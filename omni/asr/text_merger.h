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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"

namespace litert_lm::omni::asr {

// Abstract interface for aligning and merging overlapping word streams.
class TextMerger {
 public:
  // Output result of text merging for a single audio chunk.
  struct MergeResult {
    // Finalized text that will not change in subsequent chunks.
    std::string confirmed_text;
    // Pending text that is subject to alignment and change in the next chunk.
    std::string unconfirmed_text;
  };

  virtual ~TextMerger() = default;

  // Resets internal cached state for a new audio stream.
  virtual void Reset() = 0;

  // Merges curr_chunk_words into internal state and populates confirmed &
  // unconfirmed text.
  virtual absl::Status Merge(
      absl::Span<const Detokenizer::Word> curr_chunk_words,
      MergeResult* result) = 0;

  // Flushes remaining unconfirmed text at end of stream.
  virtual absl::Status Flush(MergeResult* result) = 0;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TEXT_MERGER_H_
