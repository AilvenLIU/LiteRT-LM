// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/util/runtime_debugger.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/util/safetensors_util.h"

namespace litert::lm {

namespace {

// Named constants to eliminate magic strings and numbers.
constexpr absl::string_view kMemoryCacheDir = ":memory";
constexpr absl::string_view kNoCacheDir = ":nocache";
constexpr absl::string_view kGeneratedTokensFileName = "generated_tokens.jsonl";
constexpr absl::string_view kPreEventPrefix = "pre_";
constexpr absl::string_view kPostEventPrefix = "post_";

// Serializes the content of a LiteRT tensor buffer to disk in HuggingFace
// Safetensors format using BuildSafetensorsHeader from safetensors_util.
absl::Status DumpTensorBufferToFile(absl::string_view root_directory,
                                    absl::string_view signature_name,
                                    absl::string_view tensor_name,
                                    absl::string_view event_prefix, int step,
                                    const ::litert::TensorBuffer& buffer) {
  std::error_code ec;
  std::filesystem::create_directories(std::string(root_directory), ec);
  if (ec && !std::filesystem::exists(std::string(root_directory))) {
    return absl::InternalError(
        absl::StrCat("Failed to create directory: ", root_directory));
  }

  std::string filepath =
      absl::StrCat(root_directory, "/", signature_name, "_", event_prefix,
                   tensor_name, "_step_", step, kSafetensorsExtension);

  auto packed_size_or = buffer.PackedSize();
  if (!packed_size_or.HasValue()) {
    return absl::InternalError("Unable to query tensor packed size.");
  }
  size_t bytes = *packed_size_or;

  // TensorBufferScopedLock::Create requires non-const TensorBuffer& to obtain
  // read-lock handle.
  auto lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *const_cast<::litert::TensorBuffer*>(&buffer),
      ::litert::TensorBuffer::LockMode::kRead);
  if (!lock_and_addr.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to lock tensor ", tensor_name));
  }
  void* addr = lock_and_addr->second;

  // Resolve tensor data type and dimension shape dynamically.
  absl::string_view dtype = kDefaultSafetensorsDtype;
  size_t element_size = sizeof(float);
  std::vector<int64_t> shape;

  if (auto tensor_type_or = buffer.TensorType(); tensor_type_or.HasValue()) {
    auto dtype_info = GetSafetensorsDtypeInfo(tensor_type_or->ElementType());
    dtype = dtype_info.safetensors_dtype;
    element_size = dtype_info.element_size;

    const auto& dims = tensor_type_or->Layout().Dimensions();
    if (!dims.empty()) {
      shape.assign(dims.begin(), dims.end());
    }
  }

  if (shape.empty()) {
    shape.push_back(bytes / (element_size > 0 ? element_size : 1));
  }

  std::string full_tensor_name = absl::StrCat(event_prefix, tensor_name);
  absl::flat_hash_map<std::string, std::string> metadata = {
      {"signature", std::string(signature_name)},
      {"step", absl::StrCat(step)},
  };
  std::string json_header =
      BuildSafetensorsHeader(full_tensor_name, dtype, shape, bytes, metadata);
  uint64_t header_size = json_header.size();

  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for dumping: ", filepath));
  }
  outfile.write(reinterpret_cast<const char*>(&header_size), sizeof(uint64_t));
  outfile.write(json_header.data(), json_header.size());
  outfile.write(static_cast<const char*>(addr), bytes);
  outfile.close();

  ABSL_LOG(INFO) << "Dumped safetensors tensor " << event_prefix << tensor_name
                 << " to " << filepath;
  return absl::OkStatus();
}

}  // namespace

// static
std::unique_ptr<RuntimeDebugger> RuntimeDebugger::Create(
    absl::string_view preferred_cache_dir) {
  if (preferred_cache_dir.empty() || preferred_cache_dir == kMemoryCacheDir ||
      preferred_cache_dir == kNoCacheDir) {
    ABSL_LOG(WARNING)
        << "RuntimeDebugger disabled: No valid cache directory provided "
        << "for tensor dumping (got '" << preferred_cache_dir << "').";
    return nullptr;
  }
  return std::make_unique<RuntimeDebugger>(preferred_cache_dir);
}

RuntimeDebugger::RuntimeDebugger(absl::string_view dump_dir)
    : dump_dir_(absl::StripSuffix(dump_dir, "/")) {}

LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
RuntimeDebugger::CreatePreGraphRunCallback(LlmExecutor* executor) {
  return [this, executor](
             absl::string_view signature_name, int current_step,
             absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
                 outputs) {
    const std::string* active_dir =
        cur_dump_dir_.load(std::memory_order_acquire);
    absl::string_view target_dir =
        active_dir != nullptr ? *active_dir : dump_dir_;
    DumpOutputs(target_dir, signature_name, current_step, kPreEventPrefix,
                outputs, executor);
  };
}

LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
RuntimeDebugger::CreatePostGraphRunCallback(LlmExecutor* executor) {
  return [this, executor](
             absl::string_view signature_name, int current_step,
             absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
                 outputs) {
    const std::string* active_dir =
        cur_dump_dir_.load(std::memory_order_acquire);
    absl::string_view target_dir =
        active_dir != nullptr ? *active_dir : dump_dir_;
    DumpOutputs(target_dir, signature_name, current_step, kPostEventPrefix,
                outputs, executor);
  };
}

void RuntimeDebugger::DumpOutputs(
    absl::string_view target_dir, absl::string_view signature_name,
    int current_step, absl::string_view event_prefix,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        outputs,
    LlmExecutor* executor) {
  if (target_dir.empty()) {
    return;
  }
  int step = current_step;
  if (executor != nullptr) {
    if (auto step_or = executor->GetCurrentStep(); step_or.ok()) {
      step = *step_or;
    }
  }
  for (const auto& [name, buffer] : outputs) {
    absl::Status status = DumpTensorBufferToFile(
        target_dir, signature_name, name, event_prefix, step, buffer);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Debug tensor dumping failed for " << name << ": "
                      << status;
    }
  }
}

void RuntimeDebugger::CreateSessionHandler(
    int session_id, absl::string_view preferred_cache_dir) {
  absl::string_view cache_dir =
      preferred_cache_dir.empty() ? dump_dir_ : preferred_cache_dir;
  if (cache_dir == kMemoryCacheDir || cache_dir == kNoCacheDir) {
    return;
  }
  absl::MutexLock lock(mutex_);
  session_dump_dirs_[session_id] =
      std::make_unique<std::string>(absl::StripSuffix(cache_dir, "/"));
}

void RuntimeDebugger::UnregisterSessionHandler(int session_id) {
  absl::MutexLock lock(mutex_);
  auto it = session_dump_dirs_.find(session_id);
  if (it != session_dump_dirs_.end()) {
    if (cur_dump_dir_.load(std::memory_order_relaxed) == it->second.get()) {
      cur_dump_dir_.store(nullptr, std::memory_order_release);
    }
    session_dump_dirs_.erase(it);
  }
}

void RuntimeDebugger::SwitchCurHandler(int session_id) {
  absl::MutexLock lock(mutex_);
  auto it = session_dump_dirs_.find(session_id);
  if (it != session_dump_dirs_.end()) {
    cur_dump_dir_.store(it->second.get(), std::memory_order_release);
  } else {
    cur_dump_dir_.store(nullptr, std::memory_order_release);
  }
}

void RuntimeDebugger::ObserveTokens(int session_id,
                                    const Responses& responses) {
  absl::string_view target_dir = dump_dir_;
  {
    absl::MutexLock lock(mutex_);
    auto it = session_dump_dirs_.find(session_id);
    if (it != session_dump_dirs_.end() && it->second != nullptr) {
      target_dir = *it->second;
    }
  }
  WriteTokensToFile(target_dir, responses);
}

void RuntimeDebugger::WriteTokensToFile(absl::string_view target_dir,
                                        const Responses& responses) {
  if (responses.GetTokenIds().empty() || target_dir.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(std::string(target_dir), ec);
  std::string dump_path =
      absl::StrCat(target_dir, "/", kGeneratedTokensFileName);
  std::ofstream token_file(dump_path, std::ios_base::app);
  if (!token_file.is_open()) {
    return;
  }

  nlohmann::ordered_json record;
  record["token_ids"] = responses.GetTokenIds();
  if (!responses.GetTexts().empty()) {
    record["texts"] = responses.GetTexts();
  }
  if (!responses.GetScores().empty()) {
    record["scores"] = responses.GetScores();
  }

  std::string json_line = absl::StrCat(record.dump(), "\n");
  token_file.write(json_line.data(), json_line.size());
  token_file.flush();
}

}  // namespace litert::lm
