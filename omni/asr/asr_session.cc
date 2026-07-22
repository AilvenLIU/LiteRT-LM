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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {

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
  pcm_buffer_.clear();
  mel_buffer_.clear();
  token_buffer_.clear();
  word_buffer_.clear();

  components_.preprocessor->Reset();
  components_.speech_decoder->Reset();
  components_.text_merger->Reset();
}

absl::Status AsrSession::ProcessNextChunk(TextMerger::MergeResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("MergeResult pointer is null.");
  }

  // Clear scratch buffers without deallocating capacity
  pcm_buffer_.clear();
  mel_buffer_.clear();
  token_buffer_.clear();
  word_buffer_.clear();

  LITERT_RETURN_IF_ERROR(components_.audio_source->GetNextChunk(&pcm_buffer_));
  LITERT_RETURN_IF_ERROR(
      components_.preprocessor->Preprocess(pcm_buffer_, &mel_buffer_));
  LITERT_RETURN_IF_ERROR(
      components_.speech_decoder->Decode(mel_buffer_, &token_buffer_));
  LITERT_RETURN_IF_ERROR(
      components_.detokenizer->Detokenize(token_buffer_, &word_buffer_));
  LITERT_RETURN_IF_ERROR(components_.text_merger->Merge(word_buffer_, result));

  return absl::OkStatus();
}

absl::Status AsrSession::Flush(TextMerger::MergeResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("MergeResult pointer is null.");
  }
  return components_.text_merger->Flush(result);
}

}  // namespace litert_lm::omni::asr
