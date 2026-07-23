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

#include "omni/asr/asr_session.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/detokenizer.h"
#include "omni/asr/speech_decoder.h"
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {
namespace {

#define CALL_CALLBACK_IF_ERROR(cb, status_expr) \
  do {                                          \
    const auto& _status_val = (status_expr);    \
    if (!_status_val.ok()) {                    \
      std::move(cb)(_status_val.status());      \
      return;                                   \
    }                                           \
  } while (0)

}  // namespace

absl::StatusOr<std::unique_ptr<AsrSession>> AsrSession::Create(
    Components components) {
  if (components.audio_source == nullptr) {
    return absl::InvalidArgumentError("AudioSource component is required.");
  }
  if (components.preprocessor == nullptr) {
    return absl::InvalidArgumentError(
        "AudioPreprocessor component is required.");
  }
  if (components.speech_decoder == nullptr) {
    return absl::InvalidArgumentError("SpeechDecoder component is required.");
  }
  if (components.detokenizer == nullptr) {
    return absl::InvalidArgumentError("Detokenizer component is required.");
  }
  if (components.text_merger == nullptr) {
    return absl::InvalidArgumentError("TextMerger component is required.");
  }
  return std::unique_ptr<AsrSession>(new AsrSession(std::move(components)));
}

AsrSession::AsrSession(Components components)
    : components_(std::move(components)) {}

void AsrSession::Reset() {
  components_.audio_source->Reset();
  components_.preprocessor->Reset();
  components_.speech_decoder->Reset();
  components_.detokenizer->Reset();
  components_.text_merger->Reset();
}

absl::StatusOr<TextMerger::MergeResult> AsrSession::ProcessNextChunk() {
  LITERT_ASSIGN_OR_RETURN(auto pcm_chunk,
                          components_.audio_source->GetNextChunk());
  LITERT_ASSIGN_OR_RETURN(auto mel_features,
                          components_.preprocessor->Preprocess(pcm_chunk));
  LITERT_ASSIGN_OR_RETURN(auto tokens,
                          components_.speech_decoder->Decode(mel_features));
  LITERT_ASSIGN_OR_RETURN(auto words,
                          components_.detokenizer->Detokenize(tokens));
  return components_.text_merger->Merge(words);
}

void AsrSession::ProcessNextChunkAsync(
    absl::AnyInvocable<void(absl::StatusOr<TextMerger::MergeResult>) &&>
        callback) {
  components_.audio_source->GetNextChunkAsync(
      [this, cb = std::move(callback)](
          absl::StatusOr<std::vector<float>> pcm_chunk) mutable {
        CALL_CALLBACK_IF_ERROR(cb, pcm_chunk);
        components_.preprocessor->PreprocessAsync(
            *pcm_chunk,
            [this, cb = std::move(cb)](
                absl::StatusOr<std::vector<float>> mel_features) mutable {
              CALL_CALLBACK_IF_ERROR(cb, mel_features);
              components_.speech_decoder->DecodeAsync(
                  *mel_features,
                  [this, cb = std::move(cb)](
                      absl::StatusOr<std::vector<SpeechDecoder::DecodedToken>>
                          tokens) mutable {
                    CALL_CALLBACK_IF_ERROR(cb, tokens);
                    auto words = components_.detokenizer->Detokenize(*tokens);
                    CALL_CALLBACK_IF_ERROR(cb, words);
                    std::move(cb)(components_.text_merger->Merge(*words));
                  });
            });
      });
}

absl::StatusOr<TextMerger::MergeResult> AsrSession::Flush() {
  return components_.text_merger->Flush();
}

}  // namespace litert_lm::omni::asr
