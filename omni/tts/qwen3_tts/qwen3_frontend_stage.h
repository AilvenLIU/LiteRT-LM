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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_FRONTEND_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_FRONTEND_STAGE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/tokenizer/huggingface_tokenizer.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/text_frontend.h"

namespace litert::omni::tts {

// Stage 1: Text Frontend & Prompt Framing for Qwen3-TTS
class Qwen3FrontendStage : public TextFrontend {
 public:
  Qwen3FrontendStage(Stage<std::string>* absl_nonnull text_source,
                     Qwen3StageOptions options,
                     std::shared_ptr<Environment> env = nullptr);
  ~Qwen3FrontendStage() override = default;

  absl::Status Initialize();
  void Reset() override;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);
  absl::StatusOr<std::vector<float>> EmbedText(const std::vector<int>& ids);
  absl::StatusOr<std::vector<float>> ProjectText(const std::vector<float>& rows,
                                                 int num_rows);
  absl::StatusOr<FrontendOutput> BuildPrompt(const std::string& input_text);

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;
  std::unique_ptr<support::HuggingFaceTokenizer> tokenizer_;

  // Required compiled models for text frontend stage
  std::unique_ptr<CompiledModel> text_embedding_model_;
  std::unique_ptr<CompiledModel> text_projection_model_;
  std::unique_ptr<CompiledModel> codec_embedding_model_;

  // Pre-allocated TensorBuffers initialized once to avoid runtime allocation
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;
  std::vector<TensorBuffer> text_proj_input_buffers_;
  std::vector<TensorBuffer> text_proj_output_buffers_;
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;

  // Precomputed system embeddings for fixed tokens (kTtsBos, kTtsEos, kTtsPad)
  std::vector<float> tts_bos_;
  std::vector<float> tts_eos_;
  std::vector<float> tts_pad_;

  std::vector<float> speaker_emb_;
  bool initialized_ = false;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_FRONTEND_STAGE_H_
