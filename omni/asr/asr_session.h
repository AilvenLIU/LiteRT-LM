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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_

#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/speech_decoder.h"
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {

// Orchestrates component pipeline execution for ASR speech recognition streams.
class AsrSession {
 public:
  struct Components {
    std::unique_ptr<AudioSource> audio_source;
    std::unique_ptr<AudioPreprocessor> preprocessor;
    std::unique_ptr<SpeechDecoder> speech_decoder;
    std::unique_ptr<Detokenizer> detokenizer;
    std::unique_ptr<TextMerger> text_merger;
  };

  // Creates an AsrSession instance taking ownership of configured components.
  static absl::StatusOr<std::unique_ptr<AsrSession>> Create(
      Components components);

  // Resets session and component state for a new audio stream.
  void Reset();

  // Processes the next audio chunk from AudioSource synchronously.
  // Returns absl::OutOfRangeError when audio stream ends.
  absl::StatusOr<TextMerger::MergeResult> ProcessNextChunk();

  // Processes the next audio chunk from AudioSource asynchronously.
  // Note: Callbacks can be invoked on any thread, and may be called
  // synchronously before returning on the same thread especially on error.
  // It is the caller's responsibility to synchronize resources properly.
  void ProcessNextChunkAsync(
      absl::AnyInvocable<void(absl::StatusOr<TextMerger::MergeResult>) &&>
          callback);

  // Flushes remaining unconfirmed text at stream end.
  absl::StatusOr<TextMerger::MergeResult> Flush();

 private:
  explicit AsrSession(Components components);

  Components components_;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_SESSION_H_
