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

#include "omni/tts/qwen3_tts/qwen3_frontend_stage.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "support/tokenizer/huggingface_tokenizer.h"  // from @litert
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_stage_common.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"
#include "omni/tts/text_frontend.h"

namespace litert::omni::tts {

Qwen3FrontendStage::Qwen3FrontendStage(
    Stage<std::string>* absl_nonnull text_source, Qwen3StageOptions options,
    std::shared_ptr<Environment> env)
    : TextFrontend(text_source),
      options_(std::move(options)),
      env_(std::move(env)) {}

absl::Status Qwen3FrontendStage::Initialize() {
  if (initialized_) return absl::OkStatus();

  if (env_ == nullptr) {
    return absl::InvalidArgumentError(
        "Environment is required for Qwen3FrontendStage.");
  }

  if (options_.model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view tok_buf,
                          GetAssetBuffer(options_, options_.tokenizer_file));
    ABSL_ASSIGN_OR_RETURN(
        tokenizer_,
        support::HuggingFaceTokenizer::CreateFromJson(std::string(tok_buf)));
  } else {
    std::string tok_path =
        absl::StrCat(options_.model_dir, "/", options_.tokenizer_file);
    ABSL_RETURN_IF_ERROR(CheckFileReadable(tok_path));
    ABSL_ASSIGN_OR_RETURN(
        tokenizer_, support::HuggingFaceTokenizer::CreateFromFile(tok_path));
  }

  // Load text embedding compiled model
  LITERT_ASSIGN_OR_RETURN(
      auto text_emb_model,
      CreateCompiledModel(*env_, options_, options_.text_embedding_file,
                          options_.num_threads));
  text_embedding_model_ =
      std::make_unique<CompiledModel>(std::move(text_emb_model));

  // Load text projection compiled model
  LITERT_ASSIGN_OR_RETURN(
      auto text_proj_model,
      CreateCompiledModel(*env_, options_, options_.text_projection_file,
                          options_.num_threads));
  text_projection_model_ =
      std::make_unique<CompiledModel>(std::move(text_proj_model));

  // Load codec embedding compiled model
  LITERT_ASSIGN_OR_RETURN(
      auto codec_emb_model,
      CreateCompiledModel(*env_, options_, options_.codec_embedding_file,
                          options_.num_threads));
  codec_embedding_model_ =
      std::make_unique<CompiledModel>(std::move(codec_emb_model));

  // Pre-allocate input and output TensorBuffers for each model
  LITERT_ASSIGN_OR_RETURN(text_emb_input_buffers_,
                          text_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(text_emb_output_buffers_,
                          text_embedding_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(text_proj_input_buffers_,
                          text_projection_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(text_proj_output_buffers_,
                          text_projection_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(codec_emb_input_buffers_,
                          codec_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(codec_emb_output_buffers_,
                          codec_embedding_model_->CreateOutputBuffers());

  // Precompute system embeddings for fixed tokens (kTtsBos, kTtsEos, kTtsPad)
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<float> sys_embeds,
      EmbedText({qwen3_tts::kTtsBos, qwen3_tts::kTtsEos, qwen3_tts::kTtsPad}));
  tts_bos_.assign(sys_embeds.begin(), sys_embeds.begin() + qwen3_tts::kHidden);
  tts_eos_.assign(sys_embeds.begin() + qwen3_tts::kHidden,
                  sys_embeds.begin() + 2 * qwen3_tts::kHidden);
  tts_pad_.assign(sys_embeds.begin() + 2 * qwen3_tts::kHidden,
                  sys_embeds.end());

  // Load speaker embedding from binary file array
  const std::string& spk_rel = options_.speaker_file;
  if (options_.model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view buf,
                          GetAssetBuffer(options_, spk_rel));
    if (buf.size() == qwen3_tts::kHidden * sizeof(float)) {
      speaker_emb_.resize(qwen3_tts::kHidden);
      std::memcpy(speaker_emb_.data(), buf.data(),
                  qwen3_tts::kHidden * sizeof(float));
    }
  } else {
    std::string path = absl::StrCat(options_.model_dir, "/", spk_rel);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);
      if (size == qwen3_tts::kHidden * sizeof(float)) {
        speaker_emb_.resize(qwen3_tts::kHidden);
        file.read(reinterpret_cast<char*>(speaker_emb_.data()), size);
      }
    }
  }

  if (speaker_emb_.size() != qwen3_tts::kHidden) {
    return absl::InvalidArgumentError(
        absl::StrCat("Speaker embedding size is not correct. Expected ",
                     qwen3_tts::kHidden, " but got ", speaker_emb_.size()));
  }

  initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::ProjectText(
    const std::vector<float>& rows, int num_rows) {
  std::vector<float> out(num_rows * qwen3_tts::kHidden);

  for (int r = 0; r < num_rows; ++r) {
    LITERT_RETURN_IF_ERROR(text_proj_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(rows.data() + r * 2048, 2048)));
    LITERT_RETURN_IF_ERROR(text_projection_model_->Run(
        text_proj_input_buffers_, text_proj_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        text_proj_output_buffers_[0].Read<float>(absl::MakeSpan(
            out.data() + r * qwen3_tts::kHidden, qwen3_tts::kHidden)));
  }

  return out;
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::EmbedCodecToken(
    int code_id) {
  std::vector<float> out(qwen3_tts::kHidden, 0.0f);
  if (code_id >= 0 && code_id < qwen3_tts::kCodecVocab) {
    int32_t cid = code_id;
    LITERT_RETURN_IF_ERROR(codec_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&cid, 1)));
    LITERT_RETURN_IF_ERROR(codec_embedding_model_->Run(
        codec_emb_input_buffers_, codec_emb_output_buffers_));
    LITERT_RETURN_IF_ERROR(codec_emb_output_buffers_[0].Read<float>(
        absl::MakeSpan(out.data(), qwen3_tts::kHidden)));
  }
  return out;
}

absl::StatusOr<std::vector<float>> Qwen3FrontendStage::EmbedText(
    const std::vector<int>& ids) {
  int num_ids = ids.size();
  std::vector<float> rows(num_ids * 2048);
  for (int idx = 0; idx < num_ids; ++idx) {
    ABSL_VLOG(2) << "[TRACE] EmbedText: " << ids[idx];
    int32_t tid = ids[idx];
    LITERT_RETURN_IF_ERROR(text_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&tid, 1)));
    LITERT_RETURN_IF_ERROR(text_embedding_model_->Run(
        text_emb_input_buffers_, text_emb_output_buffers_));
    LITERT_RETURN_IF_ERROR(text_emb_output_buffers_[0].Read<float>(
        absl::MakeSpan(rows.data() + idx * 2048, 2048)));
  }
  return ProjectText(rows, num_ids);
}

absl::StatusOr<FrontendOutput> Qwen3FrontendStage::BuildPrompt(
    const std::string& input_text) {
  std::string prompt_text = absl::StrCat("<|im_start|>assistant\n", input_text,
                                         "<|im_end|>\n<|im_start|>assistant\n");
  ABSL_ASSIGN_OR_RETURN(const std::vector<int> ids,
                        tokenizer_->TextToTokenIds(prompt_text));
  if (ids.size() < 8) {
    return absl::InvalidArgumentError(
        "Input text too short for tokenization framing");
  }

  std::vector<int> control;
  if (options_.language == "auto") {
    control = {qwen3_tts::kCodecNoThink, qwen3_tts::kCodecThinkBos,
               qwen3_tts::kCodecThinkEos};
  } else {
    ABSL_ASSIGN_OR_RETURN(int lang_id, GetLanguageId(options_.language));
    control = {qwen3_tts::kCodecThink, qwen3_tts::kCodecThinkBos, lang_id,
               qwen3_tts::kCodecThinkEos};
  }

  std::vector<float> codec_pre;
  auto append_codec_emb = [this, &codec_pre](int code_id) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(auto vec, EmbedCodecToken(code_id));
    codec_pre.insert(codec_pre.end(), vec.begin(), vec.end());
    return absl::OkStatus();
  };

  for (int c : control) {
    ABSL_RETURN_IF_ERROR(append_codec_emb(c));
  }
  codec_pre.insert(codec_pre.end(), speaker_emb_.begin(), speaker_emb_.end());
  ABSL_RETURN_IF_ERROR(append_codec_emb(qwen3_tts::kCodecPad));
  ABSL_RETURN_IF_ERROR(append_codec_emb(qwen3_tts::kCodecBos));

  int n_codec_pre = codec_pre.size() / qwen3_tts::kHidden;

  std::vector<int> role_ids(ids.begin(), ids.begin() + 3);
  ABSL_ASSIGN_OR_RETURN(std::vector<float> role, EmbedText(role_ids));

  std::vector<float> body((n_codec_pre - 1) * qwen3_tts::kHidden);
  for (int i = 0; i < n_codec_pre - 2; ++i) {
    std::memcpy(body.data() + i * qwen3_tts::kHidden, tts_pad_.data(),
                qwen3_tts::kHidden * sizeof(float));
  }
  std::memcpy(body.data() + (n_codec_pre - 2) * qwen3_tts::kHidden,
              tts_bos_.data(), qwen3_tts::kHidden * sizeof(float));
  for (size_t i = 0; i < body.size(); ++i) {
    body[i] += codec_pre[i];
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> first_text, EmbedText({ids[3]}));
  const float* codec_pre_last =
      codec_pre.data() + (n_codec_pre - 1) * qwen3_tts::kHidden;
  for (int i = 0; i < qwen3_tts::kHidden; ++i) {
    first_text[i] += codec_pre_last[i];
  }

  FrontendOutput out;
  out.token_ids = ids;
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), role.begin(),
                               role.end());
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), body.begin(),
                               body.end());
  out.prompt_embeddings.insert(out.prompt_embeddings.end(), first_text.begin(),
                               first_text.end());
  out.prompt_len = out.prompt_embeddings.size() / qwen3_tts::kHidden;

  std::vector<int> trailing_ids;
  if (ids.size() >= 9) {
    trailing_ids.assign(ids.begin() + 4, ids.end() - 5);
  }
  if (!trailing_ids.empty()) {
    ABSL_ASSIGN_OR_RETURN(out.trailing_embeddings, EmbedText(trailing_ids));
  }
  out.trailing_embeddings.insert(out.trailing_embeddings.end(),
                                 tts_eos_.begin(), tts_eos_.end());
  out.trailing_len = out.trailing_embeddings.size() / qwen3_tts::kHidden;
  out.tts_pad_embedding = tts_pad_;

  return out;
}

absl::Status Qwen3FrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!initialized_) {
    ABSL_RETURN_IF_ERROR(Initialize());
  }
  ABSL_VLOG(2) << "[TRACE] Starting Qwen3FrontendStage::ScheduleInternal";

  auto text_or = text_source_.GetOutput();
  if (absl::IsNotFound(text_or.status())) {
    return absl::OkStatus();
  } else if (!text_or.ok()) {
    return text_or.status();
  }
  ABSL_VLOG(2) << "[TRACE] Running BuildPrompt";
  ABSL_ASSIGN_OR_RETURN(auto prompt, BuildPrompt(*text_or));
  ABSL_VLOG(2) << "[TRACE] Finished BuildPrompt";
  PushOutput(std::move(prompt));
  return absl::OkStatus();
}

void Qwen3FrontendStage::Reset() {
  absl::MutexLock lock(mutex_);
  outputs_.clear();
}

}  // namespace litert::omni::tts
