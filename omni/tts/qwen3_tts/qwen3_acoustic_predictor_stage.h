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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_ACOUSTIC_PREDICTOR_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_ACOUSTIC_PREDICTOR_STAGE_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"

namespace litert::omni::tts {

// Stage 2: Acoustic Predictor (Talker + MTP Autoregressive RVQ Generation)
class Qwen3AcousticPredictorStage : public AcousticPredictor {
 public:
  Qwen3AcousticPredictorStage(Stage<FrontendOutput>* absl_nonnull text_frontend,
                              Qwen3StageOptions options,
                              std::shared_ptr<Environment> env);
  ~Qwen3AcousticPredictorStage() override = default;

  absl::Status Initialize();
  void Reset() override;

 protected:
  absl::Status ScheduleInternal() override;

 private:
  struct DecodeOutput {
    std::vector<float> cb0_logits;  // Shape [3072]
    std::vector<float> hidden;      // Shape [1024]
  };

  absl::Status RunPrefill(
      absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
      const std::vector<float>& prefill, int p);
  absl::StatusOr<DecodeOutput> RunDecode(
      absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
      const float* embed_1024, int pos);
  absl::StatusOr<std::vector<int>> RunMtp(const std::vector<float>& hidden,
                                          int cb0);
  int PickToken(const std::vector<float>& logits, bool do_sample);
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);
  absl::StatusOr<std::vector<float>> EmbedMtpTokens(
      const std::vector<int>& mtp_codes);

  Qwen3StageOptions options_;
  std::shared_ptr<Environment> env_;

  // Required compiled models
  std::unique_ptr<CompiledModel> talker_model_;
  std::unique_ptr<CompiledModel> mtp_model_;
  std::unique_ptr<CompiledModel> codec_embedding_model_;
  std::unique_ptr<CompiledModel> mtp_embedding_model_;

  // Pre-allocated TensorBuffers for embedding models
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;
  std::vector<TensorBuffer> mtp_emb_input_buffers_;
  std::vector<TensorBuffer> mtp_emb_output_buffers_;

  // Pre-allocated TensorBuffers for Talker prefill & decode signatures
  TensorBuffer talker_prefill_emb_buf_;
  TensorBuffer talker_prefill_pos_buf_;
  TensorBuffer talker_prefill_mask_buf_;
  absl::flat_hash_map<std::string, TensorBuffer> talker_prefill_kv_output_bufs_;

  TensorBuffer talker_decode_emb_buf_;
  TensorBuffer talker_decode_pos_buf_;
  TensorBuffer talker_decode_mask_buf_;
  TensorBuffer talker_decode_logits_buf_;
  absl::flat_hash_map<std::string, TensorBuffer> talker_decode_kv_input_bufs_;
  absl::flat_hash_map<std::string, TensorBuffer> talker_decode_kv_output_bufs_;

  // Pre-allocated TensorBuffers for MTP model
  TensorBuffer mtp_embed_buf_;
  TensorBuffer mtp_pos_buf_;
  TensorBuffer mtp_mask_buf_;
  TensorBuffer mtp_logits_out_buf_;
  absl::flat_hash_map<std::string, TensorBuffer> mtp_kv_input_bufs_;
  absl::flat_hash_map<std::string, TensorBuffer> mtp_kv_output_bufs_;

  std::vector<std::string> talker_kv_names_;
  lm::ModelSignatures mtp_signatures_;
  std::string mtp_signature_name_;
  std::string mtp_embed_input_name_;
  std::string mtp_pos_input_name_;
  std::string mtp_mask_input_name_;
  std::string mtp_logits_output_name_;
  std::vector<std::string> mtp_k_input_names_;
  std::vector<std::string> mtp_v_input_names_;
  std::vector<std::string> mtp_k_output_names_;
  std::vector<std::string> mtp_v_output_names_;
  int talker_cache_len_ = 512;
  int mtp_cache_len_ = 32;
  std::mt19937_64 rng_;
  bool initialized_ = false;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_ACOUSTIC_PREDICTOR_STAGE_H_
