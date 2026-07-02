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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_

#include <atomic>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"

namespace litert::lm {

// RuntimeDebugger manages diagnostic tracing, intermediate layer tensor
// dumping, and step-by-step token observation during model inference.
class RuntimeDebugger {
 public:
  // Creates and initializes a RuntimeDebugger instance by intelligently
  // evaluating storage paths with the following hierarchy:
  static std::unique_ptr<RuntimeDebugger> Create(
      absl::string_view preferred_cache_dir = "");

  explicit RuntimeDebugger(absl::string_view dump_dir);
  ~RuntimeDebugger() = default;

  // Returns a callback for dumping tensor buffers BEFORE each
  // computation graph execution for LiteRT compiled model executors.
  LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
  CreatePreGraphRunCallback(LlmExecutor* executor);

  // Returns a callback for dumping tensor buffers AFTER each
  // computation graph execution for LiteRT compiled model executors.
  LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
  CreatePostGraphRunCallback(LlmExecutor* executor);

  // Creates and registers a session-specific dump directory.
  void CreateSessionHandler(int session_id,
                            absl::string_view preferred_cache_dir = "");

  // Unregisters a session-specific dump directory.
  void UnregisterSessionHandler(int session_id);

  // Switches the active session dump directory for subsequent graph runs.
  void SwitchCurHandler(int session_id);

  // Observes generated tokens for a session and appends them to disk in JSONL
  // format.
  void ObserveTokens(int session_id, const Responses& responses);

  const std::string& dump_dir() const { return dump_dir_; }

 private:
  // Dumps tensor buffer outputs to disk in Safetensors format.
  void DumpOutputs(absl::string_view target_dir,
                   absl::string_view signature_name, int current_step,
                   absl::string_view event_prefix,
                   const absl::flat_hash_map<absl::string_view,
                                             ::litert::TensorBuffer>& outputs,
                   LlmExecutor* executor = nullptr);

  void WriteTokensToFile(absl::string_view target_dir,
                         const Responses& responses);

  std::string dump_dir_;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<int, std::unique_ptr<std::string>> session_dump_dirs_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<const std::string*> cur_dump_dir_{nullptr};
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_
