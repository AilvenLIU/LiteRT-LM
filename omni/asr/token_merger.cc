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

#include "omni/asr/token_merger.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/levenshtein_align.h"

namespace litert_lm::omni::asr {

void TokenMerger::Reset() { unconfirmed_words_.clear(); }

MergeResult TokenMerger::Merge(absl::Span<const std::string> curr_chunk_words) {
  MergeResult result;

  if (unconfirmed_words_.empty()) {
    // Initial chunk: cache words as unconfirmed state.
    unconfirmed_words_.assign(curr_chunk_words.begin(), curr_chunk_words.end());
    result.confirmed_text = "";
    result.unconfirmed_text = absl::StrJoin(unconfirmed_words_, " ");
    return result;
  }

  if (curr_chunk_words.empty()) {
    // Empty current chunk: confirm all cached unconfirmed words.
    result.confirmed_text = absl::StrJoin(unconfirmed_words_, " ");
    result.unconfirmed_text = "";
    unconfirmed_words_.clear();
    return result;
  }

  std::vector<AlignCode> alignment =
      AlignTokens(unconfirmed_words_, curr_chunk_words);

  size_t ref_idx = 0;
  size_t hyp_idx = 0;
  size_t first_match_ref = unconfirmed_words_.size();

  for (AlignCode code : alignment) {
    switch (code) {
      case AlignCode::kCorrect:
        if (first_match_ref == unconfirmed_words_.size()) {
          first_match_ref = ref_idx;
        }
        ref_idx++;
        hyp_idx++;
        break;
      case AlignCode::kSubstitution:
        ref_idx++;
        hyp_idx++;
        break;
      case AlignCode::kDeletion:
        ref_idx++;
        break;
      case AlignCode::kInsertion:
        hyp_idx++;
        break;
    }
  }

  if (first_match_ref < unconfirmed_words_.size()) {
    result.confirmed_text = absl::StrJoin(
        absl::MakeConstSpan(unconfirmed_words_).subspan(0, first_match_ref),
        " ");
  } else {
    // Fallback when no direct alignment is found: confirm previous context.
    result.confirmed_text = absl::StrJoin(unconfirmed_words_, " ");
  }

  unconfirmed_words_.assign(curr_chunk_words.begin(), curr_chunk_words.end());
  result.unconfirmed_text = absl::StrJoin(unconfirmed_words_, " ");

  return result;
}

MergeResult TokenMerger::Flush() {
  MergeResult result;
  result.confirmed_text = absl::StrJoin(unconfirmed_words_, " ");
  result.unconfirmed_text = "";
  unconfirmed_words_.clear();
  return result;
}

}  // namespace litert_lm::omni::asr
