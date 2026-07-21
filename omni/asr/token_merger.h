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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKEN_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKEN_MERGER_H_

#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert_lm::omni::asr {

// Output result of token merging for a single audio chunk.
struct MergeResult {
  // Finalized text that will not change in subsequent chunks.
  std::string confirmed_text;
  // Pending text that is subject to alignment and change in the next chunk.
  std::string unconfirmed_text;
};

// Aligns and merges overlapping word streams from consecutive audio chunks.
class TokenMerger {
 public:
  TokenMerger() = default;

  // Resets internal cached state for a new audio stream.
  void Reset();

  // Processes a new audio chunk's words, aligning with cached unconfirmed
  // words. Updates internal state and returns confirmed and unconfirmed text.
  MergeResult Merge(absl::Span<const std::string> curr_chunk_words);

  // Flushes all remaining unconfirmed words as confirmed text at end of stream.
  MergeResult Flush();

  // Returns cached unconfirmed words.
  absl::Span<const std::string> unconfirmed_words() const {
    return unconfirmed_words_;
  }

 private:
  std::vector<std::string> unconfirmed_words_;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TOKEN_MERGER_H_
