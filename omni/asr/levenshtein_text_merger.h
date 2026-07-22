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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {

// Aligns and merges overlapping word streams from consecutive audio chunks
// using Levenshtein distance sequence alignment.
class LevenshteinTextMerger : public TextMerger {
 public:
  LevenshteinTextMerger() = default;

  // TextMerger implementation.
  void Reset() override;
  absl::Status Merge(absl::Span<const Detokenizer::Word> curr_chunk_words,
                     MergeResult* result) override;
  absl::Status Flush(MergeResult* result) override;

  // Returns cached unconfirmed words.
  absl::Span<const std::string> unconfirmed_words() const {
    return unconfirmed_words_;
  }

 private:
  std::vector<std::string> unconfirmed_words_;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LEVENSHTEIN_TEXT_MERGER_H_
