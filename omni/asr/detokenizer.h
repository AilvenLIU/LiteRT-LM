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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/speech_decoder.h"

namespace litert_lm::omni::asr {

// Abstract interface for converting decoded tokens into words.
class Detokenizer {
 public:
  // Represents a decoded word and its optional timestamp.
  struct Word {
    std::string text;
    std::optional<int> timestamp_ms;
  };

  virtual ~Detokenizer() = default;

  // Resets internal cached state for a new audio stream.
  virtual void Reset() = 0;

  // Converts token IDs into words with timestamps.
  virtual absl::StatusOr<std::vector<Word>> Detokenize(
      absl::Span<const SpeechDecoder::DecodedToken> tokens) = 0;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_DETOKENIZER_H_
