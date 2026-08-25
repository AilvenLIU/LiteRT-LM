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

#include "schema/capabilities/capabilities.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/status_macros.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm::schema::capabilities {
namespace {

// Formats a float for the CLI report. Uses general format '%g' to preserve
// precision without trailing zeros, but appends '.0' if the formatted string
// is parsed as a whole number (i.e. does not contain a decimal point or
// scientific notation) to explicitly signal that it is a float type.
std::string FormatFloatForReport(float val) {
  std::string s = absl::StrFormat("%g", val);
  if (!absl::StrContains(s, '.') && !absl::StrContains(s, 'e')) {
    absl::StrAppend(&s, ".0");
  }
  return s;
}

// Check if the number is a magic number.
// The number is a magic number if it is prime and greater than 10.
bool IsMagicNumber(int64_t number) {
  if (number < 11) {
    return false;
  }
  if (number % 2 == 0) {
    return false;
  }
  for (int64_t i = 3; i * i <= number; i += 2) {
    if (number % i == 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<ModelCapabilities> InspectModel(std::istream& litertlm_stream) {
  litertlm_stream.seekg(0);
  LitertlmHeader header;
  ABSL_RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_stream, &header));

  const LiteRTLMMetaData* metadata = header.metadata;
  RET_CHECK_NE(metadata, nullptr);

  ModelCapabilities info;

  // 1. Discover sections and identify modality models / speculative drafters.
  const litert::lm::schema::SectionMetadata* section_metadata_obj =
      metadata->section_metadata();
  RET_CHECK_NE(section_metadata_obj, nullptr);
  auto section_objects = section_metadata_obj->objects();
  RET_CHECK_NE(section_objects, nullptr);

  bool has_llm_metadata = false;
  uint64_t llm_metadata_begin = 0;
  uint64_t llm_metadata_end = 0;

  bool found_main_tflite = false;
  uint64_t main_tflite_begin = 0;
  uint64_t main_tflite_end = 0;

  bool has_vision = false;
  bool has_audio = false;
  bool has_video = false;
  bool has_speculative_decoding = false;

  for (size_t i = 0; i < section_objects->size(); ++i) {
    const auto* section = section_objects->Get(i);
    if (section == nullptr) continue;

    if (section->data_type() == AnySectionDataType_LlmMetadataProto) {
      // LLM Metadata Protobuf section found: record its stream offsets.
      has_llm_metadata = true;
      llm_metadata_begin = section->begin_offset();
      llm_metadata_end = section->end_offset();
    } else if (section->data_type() == AnySectionDataType_TFLiteModel) {
      // Scan TFLite model attributes to detect media and speculative features.
      bool is_adapter = false;
      if (const auto* items = section->items()) {
        for (size_t j = 0; j < items->size(); ++j) {
          const KeyValuePair* item = items->Get(j);
          if (item == nullptr || item->key() == nullptr) continue;
          if (item->key()->string_view() == "model_type") {
            const auto* value = item->value_as_StringValue();
            if (value == nullptr || value->value() == nullptr) continue;
            absl::string_view model_type = value->value()->string_view();
            if (model_type == "tf_lite_vision_adapter" ||
                model_type == "tf_lite_vision_encoder") {
              has_vision = true;
              is_adapter = true;
            } else if (model_type == "tf_lite_audio_adapter" ||
                       model_type == "tf_lite_audio_encoder_hw") {
              has_audio = true;
              is_adapter = true;
            } else if (model_type == "tf_lite_video_adapter" ||
                       model_type == "tf_lite_video_encoder") {
              has_video = true;
              is_adapter = true;
            } else if (model_type == "tf_lite_mtp_drafter") {
              has_speculative_decoding = true;
              is_adapter = true;
            }
          }
        }
      }
      if (!is_adapter) {
        main_tflite_begin = section->begin_offset();
        main_tflite_end = section->end_offset();
        found_main_tflite = true;
      }
    }
  }

  // 2. Populate LlmInferenceCapability.
  LlmInferenceCapability llm_cap;
  llm_cap.input_modalities.text = true;
  llm_cap.input_modalities.vision = has_vision;
  llm_cap.input_modalities.audio = has_audio;
  llm_cap.input_modalities.video = has_video;

  llm_cap.output_modalities.text = true;
  llm_cap.supports_speculative_decoding = has_speculative_decoding;
  // Overwrite from serialized binary LlmMetadata protobuf if present.
  if (has_llm_metadata && llm_metadata_end > llm_metadata_begin) {
    size_t size = llm_metadata_end - llm_metadata_begin;
    litertlm_stream.seekg(llm_metadata_begin);
    auto buffer = std::make_unique<char[]>(size);
    litertlm_stream.read(buffer.get(), size);
    if (litertlm_stream) {
      proto::LlmMetadata proto_metadata;
      if (proto_metadata.ParseFromString(
              absl::string_view(buffer.get(), size))) {
        llm_cap.supports_thinking = proto_metadata.supports_thinking();
        llm_cap.supports_function_calling =
            proto_metadata.supports_function_calling();
        if (proto_metadata.has_sampler_params()) {
          const auto& sp = proto_metadata.sampler_params();
          llm_cap.default_sampler_params.type =
              static_cast<SamplerType>(sp.type());
          llm_cap.default_sampler_params.k = sp.k();
          llm_cap.default_sampler_params.p = sp.p();
          llm_cap.default_sampler_params.temperature = sp.temperature();
        }
        if (proto_metadata.has_llm_model_type()) {
          const auto& model_type = proto_metadata.llm_model_type();
          int max_num_patches = 0;
          // Default to 3, which is the standard default for embedding gemma v2
          // and gemma4.
          int pooling_kernel_size = 3;
          if (model_type.has_gemma4()) {
            max_num_patches = model_type.gemma4().max_num_patches();
            if (model_type.gemma4().pooling_kernel_size() > 0) {
              pooling_kernel_size = model_type.gemma4().pooling_kernel_size();
            }
          } else if (model_type.has_generic_model()) {
            max_num_patches = model_type.generic_model().max_num_patches();
            pooling_kernel_size =
                model_type.generic_model().pooling_kernel_size() > 0
                    ? model_type.generic_model().pooling_kernel_size()
                    : 1;
          } else if (model_type.has_lfm2()) {
            max_num_patches = model_type.lfm2().max_num_patches();
            pooling_kernel_size = model_type.lfm2().pooling_kernel_size() > 0
                                      ? model_type.lfm2().pooling_kernel_size()
                                      : 2;
          }
          if (max_num_patches > 0) {
            llm_cap.max_vision_token_budget =
                max_num_patches / (pooling_kernel_size * pooling_kernel_size);
          }
        }
        llm_cap.max_context_tokens = proto_metadata.max_num_tokens();
        llm_cap.is_dynamic_context = IsMagicNumber(llm_cap.max_context_tokens);
      }
    }
  }

  // 3. Inspect TFLite model for source of truth on context size.
  if (found_main_tflite && main_tflite_end > main_tflite_begin) {
    size_t size = main_tflite_end - main_tflite_begin;
    litertlm_stream.seekg(main_tflite_begin);
    auto buffer = std::make_unique<char[]>(size);
    litertlm_stream.read(buffer.get(), size);
    if (litertlm_stream) {
      const tflite::Model* tflite_model = tflite::GetModel(buffer.get());
      if (tflite_model) {
        if (tflite_model->signature_defs()) {
          for (const auto* sig : *tflite_model->signature_defs()) {
            if (sig == nullptr || sig->signature_key() == nullptr) continue;
            absl::string_view sig_key = sig->signature_key()->string_view();
            if (absl::StartsWith(sig_key, "prefill")) {
              if (sig->inputs()) {
                for (const auto* input : *sig->inputs()) {
                  if (input == nullptr || input->name() == nullptr) continue;
                  absl::string_view input_name = input->name()->string_view();
                  if (absl::StrContains(input_name, "mask")) {
                    uint32_t tensor_idx = input->tensor_index();
                    uint32_t subgraph_idx = sig->subgraph_index();
                    if (tflite_model->subgraphs() &&
                        subgraph_idx < tflite_model->subgraphs()->size()) {
                      const auto* subgraph =
                          tflite_model->subgraphs()->Get(subgraph_idx);
                      if (subgraph && tensor_idx < subgraph->tensors()->size()) {
                        const auto* tensor =
                            subgraph->tensors()->Get(tensor_idx);
                        if (tensor && tensor->shape()) {
                          int rank = tensor->shape()->size();
                          if (rank > 0) {
                            int64_t dim = tensor->shape()->Get(rank - 1);
                            llm_cap.max_context_tokens = dim;
                            llm_cap.is_dynamic_context = IsMagicNumber(dim);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  info.llm_capability = llm_cap;

  return info;
}

absl::StatusOr<ModelCapabilities> InspectModel(
    absl::string_view litertlm_path) {
  std::ifstream input_file_stream(std::string(litertlm_path), std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  return InspectModel(input_file_stream);
}

std::ostream& operator<<(std::ostream& os,
                         const SupportedModalities& modalities) {
  if (modalities.text) os << "Text ";
  if (modalities.vision) os << "Vision ";
  if (modalities.audio) os << "Audio ";
  if (modalities.video) os << "Video ";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const LlmInferenceCapability& llm_cap) {
  auto sampler_type_str = [](SamplerType type) {
    switch (type) {
      case SamplerType::kTopK:
        return "TOP_K";
      case SamplerType::kTopP:
        return "TOP_P";
      case SamplerType::kGreedy:
        return "GREEDY";
      default:
        return "UNSPECIFIED";
    }
  };

  os << "[LLM Capabilities]\n"
     << "  Supports Function Call: "
     << (llm_cap.supports_function_calling ? "YES" : "NO") << "\n"
     << "  Supports Thinking:      "
     << (llm_cap.supports_thinking ? "YES" : "NO") << "\n"
     << "  Speculative Decoding:   "
     << (llm_cap.supports_speculative_decoding ? "YES" : "NO") << "\n"
     << "  Max Vision Token Budget: " << llm_cap.max_vision_token_budget << "\n"
     << "  Sampler Type:           "
     << sampler_type_str(llm_cap.default_sampler_params.type) << "\n"
     << "  Sampler Temp:           "
     << FormatFloatForReport(llm_cap.default_sampler_params.temperature) << "\n"
     << "  Sampler Top K:          " << llm_cap.default_sampler_params.k << "\n"
     << "  Sampler Top P:          "
     << FormatFloatForReport(llm_cap.default_sampler_params.p) << "\n"
     << "  Max Context Tokens:     " << llm_cap.max_context_tokens << "\n"
     << "  Is Dynamic Context:     "
     << (llm_cap.is_dynamic_context ? "YES" : "NO") << "\n"
     << "  Input Modalities:       " << llm_cap.input_modalities << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const ModelCapabilities& capabilities) {
  if (capabilities.llm_capability.has_value()) {
    os << *capabilities.llm_capability;
  } else {
    os << "[LLM Capabilities]\n  <none>\n";
  }
  return os;
}

}  // namespace litert::lm::schema::capabilities
