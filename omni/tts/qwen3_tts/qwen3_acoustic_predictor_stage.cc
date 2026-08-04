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

#include "omni/tts/qwen3_tts/qwen3_acoustic_predictor_stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/qwen3_tts/qwen3_stage_common.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/text_frontend.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"

namespace litert::omni::tts {

Qwen3AcousticPredictorStage::Qwen3AcousticPredictorStage(
    Stage<FrontendOutput>* absl_nonnull text_frontend,
    Qwen3StageOptions options, std::shared_ptr<Environment> env)
    : AcousticPredictor(text_frontend),
      options_(std::move(options)),
      env_(std::move(env)) {
  if (options_.seed.has_value()) {
    rng_.seed(*options_.seed);
  } else {
    rng_.seed(0);
  }
}

absl::Status Qwen3AcousticPredictorStage::Initialize() {
  if (initialized_) return absl::OkStatus();

  if (!env_) {
    return absl::InvalidArgumentError(
        "Qwen3AcousticPredictorStage requires a non-null LiteRT Environment.");
  }

  // Load Talker model
  LITERT_ASSIGN_OR_RETURN(
      auto talker, CreateCompiledModel(*env_, options_, options_.talker_file,
                                       options_.num_threads,
                                       /*use_gpu=*/false));
  talker_model_ = std::make_unique<CompiledModel>(std::move(talker));

  // Load MTP model
  LITERT_ASSIGN_OR_RETURN(
      auto mtp, CreateCompiledModel(*env_, options_, options_.mtp_file,
                                    options_.num_threads,
                                    /*use_gpu=*/false));
  mtp_model_ = std::make_unique<CompiledModel>(std::move(mtp));

  // Load Codec embedding model
  LITERT_ASSIGN_OR_RETURN(
      auto codec_emb,
      CreateCompiledModel(*env_, options_, options_.codec_embedding_file,
                          options_.num_threads));
  codec_embedding_model_ =
      std::make_unique<CompiledModel>(std::move(codec_emb));

  // Load MTP embedding model
  LITERT_ASSIGN_OR_RETURN(
      auto mtp_emb,
      CreateCompiledModel(*env_, options_, options_.mtp_embedding_file,
                          options_.num_threads));
  mtp_embedding_model_ = std::make_unique<CompiledModel>(std::move(mtp_emb));

  // Pre-allocate embedding model TensorBuffers
  LITERT_ASSIGN_OR_RETURN(codec_emb_input_buffers_,
                          codec_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(codec_emb_output_buffers_,
                          codec_embedding_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(mtp_emb_input_buffers_,
                          mtp_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(mtp_emb_output_buffers_,
                          mtp_embedding_model_->CreateOutputBuffers());

  // Set up MTP signature and input/output names
  LITERT_ASSIGN_OR_RETURN(auto decode_signature, mtp_model_->GetSignature(0));
  mtp_signature_name_ = std::string(decode_signature.Key());

  ABSL_ASSIGN_OR_RETURN(
      mtp_signatures_,
      lm::GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                                 decode_signature.OutputNames(),
                                                 /*strict=*/false));

  std::string mtp_k_root_name;
  std::string mtp_v_root_name;
  ABSL_RETURN_IF_ERROR(lm::GetKVCacheRootNames(
      decode_signature.InputNames(), decode_signature.OutputNames(),
      mtp_k_root_name, mtp_v_root_name));

  mtp_k_input_names_.clear();
  mtp_v_input_names_.clear();
  mtp_k_output_names_.clear();
  mtp_v_output_names_.clear();

  for (absl::string_view name : decode_signature.InputNames()) {
    if (absl::StartsWith(name, mtp_k_root_name)) {
      mtp_k_input_names_.push_back(std::string(name));
    } else if (absl::StartsWith(name, mtp_v_root_name)) {
      mtp_v_input_names_.push_back(std::string(name));
    }
  }
  for (absl::string_view name : decode_signature.OutputNames()) {
    if (absl::StartsWith(name, mtp_k_root_name)) {
      mtp_k_output_names_.push_back(std::string(name));
    } else if (absl::StartsWith(name, mtp_v_root_name)) {
      mtp_v_output_names_.push_back(std::string(name));
    }
  }

  if (mtp_signatures_.input_embeddings.has_value() &&
      !mtp_signatures_.input_embeddings->empty()) {
    mtp_embed_input_name_ = *mtp_signatures_.input_embeddings;
  } else if (!mtp_signatures_.input_tokens.empty()) {
    mtp_embed_input_name_ = mtp_signatures_.input_tokens;
  } else if (!decode_signature.InputNames().empty()) {
    mtp_embed_input_name_ = std::string(decode_signature.InputNames()[0]);
  }

  if (!mtp_signatures_.input_positions.empty()) {
    mtp_pos_input_name_ = mtp_signatures_.input_positions;
  } else if (decode_signature.InputNames().size() > 1) {
    mtp_pos_input_name_ = std::string(decode_signature.InputNames()[1]);
  }

  if (mtp_signatures_.input_attn_mask.has_value() &&
      !mtp_signatures_.input_attn_mask->empty()) {
    mtp_mask_input_name_ = *mtp_signatures_.input_attn_mask;
  } else if (decode_signature.InputNames().size() > 2) {
    mtp_mask_input_name_ = std::string(decode_signature.InputNames()[2]);
  }

  if (!mtp_signatures_.output_logits.empty()) {
    mtp_logits_output_name_ = mtp_signatures_.output_logits;
  } else if (!decode_signature.OutputNames().empty()) {
    mtp_logits_output_name_ = std::string(decode_signature.OutputNames()[0]);
  }

  // Pre-allocate MTP TensorBuffers
  LITERT_ASSIGN_OR_RETURN(mtp_embed_buf_,
                          mtp_model_->CreateInputBuffer(mtp_signature_name_,
                                                        mtp_embed_input_name_));
  LITERT_ASSIGN_OR_RETURN(
      mtp_pos_buf_,
      mtp_model_->CreateInputBuffer(mtp_signature_name_, mtp_pos_input_name_));
  LITERT_ASSIGN_OR_RETURN(
      mtp_mask_buf_,
      mtp_model_->CreateInputBuffer(mtp_signature_name_, mtp_mask_input_name_));
  LITERT_ASSIGN_OR_RETURN(mtp_logits_out_buf_,
                          mtp_model_->CreateOutputBuffer(
                              mtp_signature_name_, mtp_logits_output_name_));

  for (const auto& k_name : mtp_k_input_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto buf, mtp_model_->CreateInputBuffer(mtp_signature_name_, k_name));
    mtp_kv_input_bufs_[k_name] = std::move(buf);
  }
  for (const auto& v_name : mtp_v_input_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto buf, mtp_model_->CreateInputBuffer(mtp_signature_name_, v_name));
    mtp_kv_input_bufs_[v_name] = std::move(buf);
  }
  for (const auto& out_name : mtp_k_output_names_) {
    LITERT_ASSIGN_OR_RETURN(auto buf, mtp_model_->CreateOutputBuffer(
                                          mtp_signature_name_, out_name));
    mtp_kv_output_bufs_[out_name] = std::move(buf);
  }
  for (const auto& out_name : mtp_v_output_names_) {
    LITERT_ASSIGN_OR_RETURN(auto buf, mtp_model_->CreateOutputBuffer(
                                          mtp_signature_name_, out_name));
    mtp_kv_output_bufs_[out_name] = std::move(buf);
  }

  // Set up Talker model signature and pre-allocate TensorBuffers
  LITERT_ASSIGN_OR_RETURN(auto decode_inputs,
                          talker_model_->GetSignatureInputNames("decode"));
  for (auto input_name : decode_inputs) {
    if (absl::StartsWith(input_name, "kv_cache")) {
      talker_kv_names_.emplace_back(input_name);
    }
  }

  // Pre-allocate Talker prefill buffers
  LITERT_ASSIGN_OR_RETURN(
      talker_prefill_emb_buf_,
      talker_model_->CreateInputBuffer("prefill_32", "embeddings"));
  LITERT_ASSIGN_OR_RETURN(
      talker_prefill_pos_buf_,
      talker_model_->CreateInputBuffer("prefill_32", "input_pos"));
  LITERT_ASSIGN_OR_RETURN(
      talker_prefill_mask_buf_,
      talker_model_->CreateInputBuffer("prefill_32", "mask"));
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto out_kv, talker_model_->CreateOutputBuffer("prefill_32", kv_name));
    talker_prefill_kv_output_bufs_[kv_name] = std::move(out_kv);
  }

  // Pre-allocate Talker decode buffers
  LITERT_ASSIGN_OR_RETURN(
      talker_decode_emb_buf_,
      talker_model_->CreateInputBuffer("decode", "embeddings"));
  LITERT_ASSIGN_OR_RETURN(
      talker_decode_pos_buf_,
      talker_model_->CreateInputBuffer("decode", "input_pos"));
  LITERT_ASSIGN_OR_RETURN(talker_decode_mask_buf_,
                          talker_model_->CreateInputBuffer("decode", "mask"));
  LITERT_ASSIGN_OR_RETURN(
      talker_decode_logits_buf_,
      talker_model_->CreateOutputBuffer("decode", "logits"));
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto in_kv, talker_model_->CreateInputBuffer("decode", kv_name));
    talker_decode_kv_input_bufs_[kv_name] = std::move(in_kv);
    LITERT_ASSIGN_OR_RETURN(
        auto out_kv, talker_model_->CreateOutputBuffer("decode", kv_name));
    talker_decode_kv_output_bufs_[kv_name] = std::move(out_kv);
  }

  talker_cache_len_ = 512;
  LITERT_ASSIGN_OR_RETURN(auto type, talker_decode_mask_buf_.TensorType());
  auto dims = type.Layout().Dimensions();
  if (dims.size() > 3) {
    talker_cache_len_ = dims[3];
  }

  initialized_ = true;
  return absl::OkStatus();
}

// TODO b/538727793: use cpu/gpu sampler from litert-lm.
int Qwen3AcousticPredictorStage::PickToken(const std::vector<float>& logits,
                                           bool do_sample) {
  if (!do_sample) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
  }

  int n = logits.size();
  std::vector<double> scaled(n);
  double temp = std::max(static_cast<double>(options_.temperature), 1e-6);
  double max_logit = -1e30;

  for (int i = 0; i < n; ++i) {
    scaled[i] = static_cast<double>(logits[i]) / temp;
    if (scaled[i] > max_logit) max_logit = scaled[i];
  }

  int top_k = std::min(options_.top_k, n);
  std::vector<std::pair<double, int>> pairs(n);
  for (int i = 0; i < n; ++i) {
    pairs[i] = {scaled[i], i};
  }
  absl::c_partial_sort(
      pairs, pairs.begin() + top_k,
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<double> probs(top_k);
  double sum_p = 0.0;
  for (int i = 0; i < top_k; ++i) {
    probs[i] = std::exp(pairs[i].first - max_logit);
    sum_p += probs[i];
  }
  for (int i = 0; i < top_k; ++i) {
    probs[i] /= sum_p;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return pairs[dist(rng_)].second;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(qwen3_tts::kHidden, 0.0f);
  LITERT_RETURN_IF_ERROR(codec_emb_input_buffers_[0].Write<int32_t>(
      absl::MakeConstSpan(&code_id, 1)));
  LITERT_RETURN_IF_ERROR(codec_embedding_model_->Run(
      codec_emb_input_buffers_, codec_emb_output_buffers_));
  LITERT_RETURN_IF_ERROR(
      codec_emb_output_buffers_[0].Read<float>(absl::MakeSpan(out)));
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3AcousticPredictorStage::EmbedMtpTokens(
    const std::vector<int>& mtp_codes) {
  std::vector<float> sum_embed(qwen3_tts::kHidden, 0.0f);
  int num_codes = mtp_codes.size();

  for (int g = 0; g < num_codes; ++g) {
    int32_t global_id = g * 2048 + mtp_codes[g];
    LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&global_id, 1)));

    LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(mtp_emb_input_buffers_,
                                                     mtp_emb_output_buffers_));
    LITERT_ASSIGN_OR_RETURN(auto copy_res, support::CopyFromTensorBuffer<float>(
                                               mtp_emb_output_buffers_[0]));
    absl::c_transform(copy_res, sum_embed, sum_embed.begin(),
                      [](float a, float b) { return a + b; });
  }
  return sum_embed;
}

absl::Status Qwen3AcousticPredictorStage::RunPrefill(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
    const std::vector<float>& prefill, int p) {
  if (p > 32) {
    return absl::InvalidArgumentError(
        absl::StrCat("Prompt too long for prefill_32: ", p));
  }

  std::vector<float> emb_32(32 * 1024, 0.0f);
  std::memcpy(emb_32.data(), prefill.data(), p * 1024 * sizeof(float));
  LITERT_RETURN_IF_ERROR(
      talker_prefill_emb_buf_.Write<float>(absl::MakeConstSpan(emb_32)));

  std::vector<int32_t> pos_32(32);
  for (int i = 0; i < 32; ++i) pos_32[i] = i;
  LITERT_RETURN_IF_ERROR(
      talker_prefill_pos_buf_.Write<int32_t>(absl::MakeConstSpan(pos_32)));

  std::vector<float> mask_32(32 * talker_cache_len_, qwen3_tts::kNegInf);
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j <= i && j < talker_cache_len_; ++j) {
      mask_32[i * talker_cache_len_ + j] = 0.0f;
    }
  }
  LITERT_RETURN_IF_ERROR(
      talker_prefill_mask_buf_.Write<float>(absl::MakeConstSpan(mask_32)));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  LITERT_ASSIGN_OR_RETURN(auto emb_dup, talker_prefill_emb_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto pos_dup, talker_prefill_pos_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto mask_dup, talker_prefill_mask_buf_.Duplicate());
  input_map.insert_or_assign("embeddings", std::move(emb_dup));
  input_map.insert_or_assign("input_pos", std::move(pos_dup));
  input_map.insert_or_assign("mask", std::move(mask_dup));

  for (const auto& kv_name : talker_kv_names_) {
    if (kv_buffers.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(auto dup, kv_buffers[kv_name].Duplicate());
      input_map.insert_or_assign(kv_name, std::move(dup));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          size_t size_bytes,
          talker_decode_kv_input_bufs_[kv_name].PackedSize());
      if (size_bytes > 0) {
        std::vector<float> zeros(size_bytes / sizeof(float), 0.0f);
        LITERT_RETURN_IF_ERROR(
            talker_decode_kv_input_bufs_[kv_name].Write<float>(
                absl::MakeConstSpan(zeros)));
      }
      LITERT_ASSIGN_OR_RETURN(
          auto dup, talker_decode_kv_input_bufs_[kv_name].Duplicate());
      input_map.insert_or_assign(kv_name, std::move(dup));
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto dup, talker_prefill_kv_output_bufs_[kv_name].Duplicate());
    output_map.insert_or_assign(kv_name, std::move(dup));
  }

  LITERT_RETURN_IF_ERROR(
      talker_model_->Run("prefill_32", input_map, output_map));

  for (const auto& kv_name : talker_kv_names_) {
    if (output_map.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         output_map[kv_name], TensorBuffer::LockMode::kRead));
      if (lock.second == nullptr) {
        return absl::InternalError(absl::StrCat("Null lock for ", kv_name));
      }
      LITERT_ASSIGN_OR_RETURN(size_t bytes, output_map[kv_name].Size());
      size_t num_floats = bytes / sizeof(float);

      LITERT_ASSIGN_OR_RETURN(
          size_t dec_bytes, talker_decode_kv_input_bufs_[kv_name].PackedSize());
      size_t total_dec_floats = dec_bytes / sizeof(float);
      std::vector<float> dec_buffer(total_dec_floats, 0.0f);
      std::memcpy(dec_buffer.data(), lock.second,
                  std::min(num_floats, total_dec_floats) * sizeof(float));
      LITERT_RETURN_IF_ERROR(talker_decode_kv_input_bufs_[kv_name].Write<float>(
          absl::MakeConstSpan(dec_buffer)));
      LITERT_ASSIGN_OR_RETURN(
          auto dup, talker_decode_kv_input_bufs_[kv_name].Duplicate());
      kv_buffers.insert_or_assign(kv_name, std::move(dup));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Qwen3AcousticPredictorStage::DecodeOutput>
Qwen3AcousticPredictorStage::RunDecode(
    absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_buffers,
    const float* embed_1024, int pos) {
  LITERT_RETURN_IF_ERROR(talker_decode_emb_buf_.Write<float>(
      absl::MakeConstSpan(embed_1024, qwen3_tts::kHidden)));

  int32_t pos_val = pos;
  LITERT_RETURN_IF_ERROR(
      talker_decode_pos_buf_.Write<int32_t>(absl::MakeConstSpan(&pos_val, 1)));

  std::vector<float> mask(talker_cache_len_, qwen3_tts::kNegInf);
  for (int j = 0; j <= pos && j < talker_cache_len_; ++j) {
    mask[j] = 0.0f;
  }
  LITERT_RETURN_IF_ERROR(
      talker_decode_mask_buf_.Write<float>(absl::MakeConstSpan(mask)));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  LITERT_ASSIGN_OR_RETURN(auto emb_dup, talker_decode_emb_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto pos_dup, talker_decode_pos_buf_.Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto mask_dup, talker_decode_mask_buf_.Duplicate());
  input_map.insert_or_assign("embeddings", std::move(emb_dup));
  input_map.insert_or_assign("input_pos", std::move(pos_dup));
  input_map.insert_or_assign("mask", std::move(mask_dup));

  for (const auto& kv_name : talker_kv_names_) {
    if (kv_buffers.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(auto dup, kv_buffers[kv_name].Duplicate());
      input_map.insert_or_assign(kv_name, std::move(dup));
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  LITERT_ASSIGN_OR_RETURN(auto logits_dup,
                          talker_decode_logits_buf_.Duplicate());
  output_map.insert_or_assign("logits", std::move(logits_dup));
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(auto dup,
                            talker_decode_kv_output_bufs_[kv_name].Duplicate());
    output_map.insert_or_assign(kv_name, std::move(dup));
  }

  LITERT_RETURN_IF_ERROR(talker_model_->Run("decode", input_map, output_map));

  for (const auto& kv_name : talker_kv_names_) {
    if (output_map.contains(kv_name)) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         output_map[kv_name], TensorBuffer::LockMode::kRead));
      if (lock.second == nullptr) {
        return absl::InternalError(absl::StrCat("Null lock for ", kv_name));
      }
      LITERT_ASSIGN_OR_RETURN(size_t bytes, output_map[kv_name].Size());
      LITERT_RETURN_IF_ERROR(talker_decode_kv_input_bufs_[kv_name].Write<float>(
          absl::MakeConstSpan(lock.second, bytes / sizeof(float))));
      LITERT_ASSIGN_OR_RETURN(
          auto dup, talker_decode_kv_input_bufs_[kv_name].Duplicate());
      kv_buffers.insert_or_assign(kv_name, std::move(dup));
    }
  }

  DecodeOutput out;
  out.cb0_logits.resize(qwen3_tts::kCodecVocab);
  out.hidden.resize(qwen3_tts::kHidden);

  {
    LITERT_ASSIGN_OR_RETURN(
        auto lock,
        TensorBufferScopedLock::Create<const float>(
            talker_decode_logits_buf_, TensorBuffer::LockMode::kRead));
    if (lock.second == nullptr) {
      return absl::InternalError("Null decode logits output buffer");
    }
    const float* raw_logits = lock.second;
    std::memcpy(out.cb0_logits.data(), raw_logits,
                qwen3_tts::kCodecVocab * sizeof(float));
    std::memcpy(out.hidden.data(), raw_logits + qwen3_tts::kCodecVocab,
                qwen3_tts::kHidden * sizeof(float));
  }

  return out;
}

absl::StatusOr<std::vector<int>> Qwen3AcousticPredictorStage::RunMtp(
    const std::vector<float>& hidden, int cb0) {
  constexpr int cache_floats_per_arg = 8 * 32 * 128;
  std::vector<float> zeros(cache_floats_per_arg, 0.0f);

  absl::flat_hash_map<absl::string_view, TensorBuffer> kv_input_buffers;
  for (const auto& k_name : mtp_k_input_names_) {
    LITERT_RETURN_IF_ERROR(
        mtp_kv_input_bufs_[k_name].Write<float>(absl::MakeConstSpan(zeros)));
    LITERT_ASSIGN_OR_RETURN(auto dup, mtp_kv_input_bufs_[k_name].Duplicate());
    kv_input_buffers.insert_or_assign(k_name, std::move(dup));
  }
  for (const auto& v_name : mtp_v_input_names_) {
    LITERT_RETURN_IF_ERROR(
        mtp_kv_input_bufs_[v_name].Write<float>(absl::MakeConstSpan(zeros)));
    LITERT_ASSIGN_OR_RETURN(auto dup, mtp_kv_input_bufs_[v_name].Duplicate());
    kv_input_buffers.insert_or_assign(v_name, std::move(dup));
  }

  std::vector<int> codes;
  codes.reserve(qwen3_tts::kNumCodeGroups - 1);

  for (int t = 0; t < qwen3_tts::kNumCodeGroups - 1; ++t) {
    std::vector<float> embed(qwen3_tts::kHidden, 0.0f);
    if (t == 0) {
      embed = hidden;
    } else if (t == 1) {
      ABSL_ASSIGN_OR_RETURN(embed, EmbedCodecToken(cb0));
    } else {
      int head_idx = t - 2;
      int prev_code = codes[head_idx];
      int32_t global_id = head_idx * 2048 + prev_code;
      LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
          absl::MakeConstSpan(&global_id, 1)));
      LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(
          mtp_emb_input_buffers_, mtp_emb_output_buffers_));
      LITERT_RETURN_IF_ERROR(mtp_emb_output_buffers_[0].Read<float>(
          absl::MakeSpan(embed.data(), qwen3_tts::kHidden)));
    }

    LITERT_RETURN_IF_ERROR(
        mtp_embed_buf_.Write<float>(absl::MakeConstSpan(embed)));
    int32_t t_val = t;
    LITERT_RETURN_IF_ERROR(
        mtp_pos_buf_.Write<int32_t>(absl::MakeConstSpan(&t_val, 1)));
    std::vector<float> mask(mtp_cache_len_, qwen3_tts::kNegInf);
    for (int j = 0; j <= t && j < mtp_cache_len_; ++j) mask[j] = 0.0f;
    LITERT_RETURN_IF_ERROR(
        mtp_mask_buf_.Write<float>(absl::MakeConstSpan(mask)));

    absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
    LITERT_ASSIGN_OR_RETURN(auto embed_dup, mtp_embed_buf_.Duplicate());
    LITERT_ASSIGN_OR_RETURN(auto pos_dup, mtp_pos_buf_.Duplicate());
    LITERT_ASSIGN_OR_RETURN(auto mask_dup, mtp_mask_buf_.Duplicate());
    input_map.insert_or_assign(mtp_embed_input_name_, std::move(embed_dup));
    input_map.insert_or_assign(mtp_pos_input_name_, std::move(pos_dup));
    input_map.insert_or_assign(mtp_mask_input_name_, std::move(mask_dup));
    for (const auto& [name, buf] : kv_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(auto dup, buf.Duplicate());
      input_map.insert_or_assign(name, std::move(dup));
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer> step_output_map;
    LITERT_ASSIGN_OR_RETURN(auto logits_dup, mtp_logits_out_buf_.Duplicate());
    step_output_map.insert_or_assign(mtp_logits_output_name_,
                                     std::move(logits_dup));

    for (const auto& out_name : mtp_k_output_names_) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_output_bufs_[out_name].Duplicate());
      step_output_map.insert_or_assign(out_name, std::move(dup));
    }
    for (const auto& out_name : mtp_v_output_names_) {
      LITERT_ASSIGN_OR_RETURN(auto dup,
                              mtp_kv_output_bufs_[out_name].Duplicate());
      step_output_map.insert_or_assign(out_name, std::move(dup));
    }

    LITERT_RETURN_IF_ERROR(
        mtp_model_->Run(mtp_signature_name_, input_map, step_output_map));

    kv_input_buffers.clear();
    for (size_t i = 0;
         i < mtp_k_output_names_.size() && i < mtp_k_input_names_.size(); ++i) {
      const auto& out_name = mtp_k_output_names_[i];
      const auto& next_in_name = mtp_k_input_names_[i];
      if (step_output_map.contains(out_name)) {
        LITERT_ASSIGN_OR_RETURN(
            auto lock,
            TensorBufferScopedLock::Create<const float>(
                step_output_map[out_name], TensorBuffer::LockMode::kRead));
        if (lock.second == nullptr) {
          return absl::InternalError(absl::StrCat("Null lock for ", out_name));
        }
        LITERT_ASSIGN_OR_RETURN(size_t bytes, step_output_map[out_name].Size());
        LITERT_RETURN_IF_ERROR(mtp_kv_input_bufs_[next_in_name].Write<float>(
            absl::MakeConstSpan(lock.second, bytes / sizeof(float))));
        LITERT_ASSIGN_OR_RETURN(auto dup,
                                mtp_kv_input_bufs_[next_in_name].Duplicate());
        kv_input_buffers.insert_or_assign(next_in_name, std::move(dup));
      }
    }
    for (size_t i = 0;
         i < mtp_v_output_names_.size() && i < mtp_v_input_names_.size(); ++i) {
      const auto& out_name = mtp_v_output_names_[i];
      const auto& next_in_name = mtp_v_input_names_[i];
      if (step_output_map.contains(out_name)) {
        LITERT_ASSIGN_OR_RETURN(
            auto lock,
            TensorBufferScopedLock::Create<const float>(
                step_output_map[out_name], TensorBuffer::LockMode::kRead));
        if (lock.second == nullptr) {
          return absl::InternalError(absl::StrCat("Null lock for ", out_name));
        }
        LITERT_ASSIGN_OR_RETURN(size_t bytes, step_output_map[out_name].Size());
        LITERT_RETURN_IF_ERROR(mtp_kv_input_bufs_[next_in_name].Write<float>(
            absl::MakeConstSpan(lock.second, bytes / sizeof(float))));
        LITERT_ASSIGN_OR_RETURN(auto dup,
                                mtp_kv_input_bufs_[next_in_name].Duplicate());
        kv_input_buffers.insert_or_assign(next_in_name, std::move(dup));
      }
    }

    if (t >= 1) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         mtp_logits_out_buf_, TensorBuffer::LockMode::kRead));
      if (lock.second == nullptr) {
        return absl::InternalError("Null MTP logits output buffer");
      }
      int head_idx = t - 1;
      const float* logits_ptr = lock.second + head_idx * 2048;
      std::vector<float> logits(logits_ptr, logits_ptr + 2048);
      int picked_code = PickToken(logits, options_.do_sample);
      codes.push_back(picked_code);
    }
  }

  return codes;
}

absl::Status Qwen3AcousticPredictorStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!initialized_) {
    ABSL_RETURN_IF_ERROR(Initialize());
  }
  ABSL_VLOG(2)
      << "[TRACE] Starting Qwen3AcousticPredictorStage::ScheduleInternal";

  auto frontend_or = text_frontend_.GetOutput();
  if (absl::IsNotFound(frontend_or.status())) {
    return absl::OkStatus();
  } else if (!frontend_or.ok()) {
    return frontend_or.status();
  }
  const auto& frontend = *frontend_or;

  absl::flat_hash_map<absl::string_view, TensorBuffer> kv_buffers;
  for (const auto& kv_name : talker_kv_names_) {
    LITERT_ASSIGN_OR_RETURN(size_t size_bytes,
                            talker_decode_kv_input_bufs_[kv_name].PackedSize());
    if (size_bytes > 0) {
      std::vector<float> zeros(size_bytes / sizeof(float), 0.0f);
      LITERT_RETURN_IF_ERROR(talker_decode_kv_input_bufs_[kv_name].Write<float>(
          absl::MakeConstSpan(zeros)));
    }
    LITERT_ASSIGN_OR_RETURN(auto dup,
                            talker_decode_kv_input_bufs_[kv_name].Duplicate());
    kv_buffers.insert_or_assign(kv_name, std::move(dup));
  }

  Qwen3AcousticPredictorStage::DecodeOutput decode_out;
  for (int pos = 0; pos < frontend.prompt_len; ++pos) {
    const float* emb_ptr = frontend.prompt_embeddings.data() + pos * 1024;
    ABSL_VLOG(2) << "[TRACE] Running decode for prompt pos=" << pos;
    ABSL_ASSIGN_OR_RETURN(decode_out, RunDecode(kv_buffers, emb_ptr, pos));
  }

  int pos = frontend.prompt_len - 1;

  std::vector<float> suppress(qwen3_tts::kCodecVocab, 0.0f);
  for (int i = 2048; i < qwen3_tts::kCodecVocab; ++i) {
    suppress[i] = qwen3_tts::kNegInf;
  }
  suppress[qwen3_tts::kCodecEos] = 0.0f;

  std::vector<std::vector<int>> frames;
  std::vector<float> codec_features;
  absl::flat_hash_set<int> history;

  while (static_cast<int>(frames.size()) < options_.max_frames) {
    std::vector<float> scores(qwen3_tts::kCodecVocab);
    for (int i = 0; i < qwen3_tts::kCodecVocab; ++i) {
      scores[i] = decode_out.cb0_logits[i] + suppress[i];
    }

    if (frames.size() < 2) {
      scores[qwen3_tts::kCodecEos] = qwen3_tts::kNegInf;
    }

    for (int token : history) {
      if (scores[token] > 0) {
        scores[token] /= options_.repetition_penalty;
      } else {
        scores[token] *= options_.repetition_penalty;
      }
    }

    int cb0 = PickToken(scores, options_.do_sample);
    history.insert(cb0);
    ABSL_VLOG(2) << "[TRACE] Generated frame " << frames.size()
                 << " with cb0=" << cb0;
    if (cb0 == qwen3_tts::kCodecEos) break;

    ABSL_ASSIGN_OR_RETURN(auto mtp_codes, RunMtp(decode_out.hidden, cb0));
    std::vector<int> frame;
    frame.reserve(qwen3_tts::kNumCodeGroups);
    frame.push_back(cb0);
    frame.insert(frame.end(), mtp_codes.begin(), mtp_codes.end());
    frames.push_back(std::move(frame));

    ABSL_ASSIGN_OR_RETURN(auto codec_vec, EmbedCodecToken(cb0));
    ABSL_ASSIGN_OR_RETURN(auto mtp_vec, EmbedMtpTokens(mtp_codes));

    std::vector<float> embed(qwen3_tts::kHidden);
    for (int i = 0; i < qwen3_tts::kHidden; ++i) {
      float val = codec_vec[i] + mtp_vec[i];
      embed[i] = val;
      codec_features.push_back(val);
    }

    int step = frames.size() - 1;
    if (step < frontend.trailing_len) {
      const float* tr_ptr =
          frontend.trailing_embeddings.data() + step * qwen3_tts::kHidden;
      for (int i = 0; i < qwen3_tts::kHidden; ++i) embed[i] += tr_ptr[i];
    } else {
      for (int i = 0; i < qwen3_tts::kHidden; ++i)
        embed[i] += frontend.tts_pad_embedding[i];
    }

    pos += 1;
    if (pos >= talker_cache_len_) {
      ABSL_LOG(WARNING) << "Reached talker_cache_len_ (" << talker_cache_len_
                        << "), stopping decode generation.";
      break;
    }
    ABSL_ASSIGN_OR_RETURN(decode_out, RunDecode(kv_buffers, embed.data(), pos));
  }

  AcousticOutput out;
  out.rvq_frames = std::move(frames);
  out.codec_features = std::move(codec_features);
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void Qwen3AcousticPredictorStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
}

}  // namespace litert::omni::tts
