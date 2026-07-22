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

#include "omni/asr/levenshtein_text_merger.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "omni/asr/detokenizer.h"
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {
namespace {

std::vector<Detokenizer::Word> ToWords(
    const std::vector<std::string>& strings) {
  std::vector<Detokenizer::Word> words;
  words.reserve(strings.size());
  for (const auto& s : strings) {
    words.push_back({s, -1});
  }
  return words;
}

TEST(LevenshteinTextMergerTest, InitialChunkReturnsUnconfirmedOnly) {
  LevenshteinTextMerger merger;
  auto curr_chunk = ToWords({"hello", "world"});

  TextMerger::MergeResult result;
  ASSERT_TRUE(merger.Merge(curr_chunk, &result).ok());

  EXPECT_EQ(result.confirmed_text, "");
  EXPECT_EQ(result.unconfirmed_text, "hello world");
}

TEST(LevenshteinTextMergerTest, SequentialMergeFlow) {
  LevenshteinTextMerger merger;

  // Chunk 1: "hello world this is"
  TextMerger::MergeResult res1;
  ASSERT_TRUE(
      merger.Merge(ToWords({"hello", "world", "this", "is"}), &res1).ok());
  EXPECT_EQ(res1.confirmed_text, "");
  EXPECT_EQ(res1.unconfirmed_text, "hello world this is");

  // Chunk 2: overlaps at "this is", adds "a test"
  TextMerger::MergeResult res2;
  ASSERT_TRUE(merger.Merge(ToWords({"this", "is", "a", "test"}), &res2).ok());
  EXPECT_EQ(res2.confirmed_text, "hello world");
  EXPECT_EQ(res2.unconfirmed_text, "this is a test");

  // Chunk 3: overlaps at "a test", adds "of streaming"
  TextMerger::MergeResult res3;
  ASSERT_TRUE(
      merger.Merge(ToWords({"a", "test", "of", "streaming"}), &res3).ok());
  EXPECT_EQ(res3.confirmed_text, "this is");
  EXPECT_EQ(res3.unconfirmed_text, "a test of streaming");

  // End of stream flush
  TextMerger::MergeResult res_flush;
  ASSERT_TRUE(merger.Flush(&res_flush).ok());
  EXPECT_EQ(res_flush.confirmed_text, "a test of streaming");
  EXPECT_EQ(res_flush.unconfirmed_text, "");
}

TEST(LevenshteinTextMergerTest, ResetClearsState) {
  LevenshteinTextMerger merger;

  TextMerger::MergeResult dummy_res;
  EXPECT_TRUE(merger.Merge(ToWords({"hello", "world"}), &dummy_res).ok());
  EXPECT_FALSE(merger.unconfirmed_words().empty());

  merger.Reset();
  EXPECT_TRUE(merger.unconfirmed_words().empty());

  // First merge after reset behaves as initial chunk
  TextMerger::MergeResult result;
  ASSERT_TRUE(merger.Merge(ToWords({"new", "stream"}), &result).ok());
  EXPECT_EQ(result.confirmed_text, "");
  EXPECT_EQ(result.unconfirmed_text, "new stream");
}

TEST(LevenshteinTextMergerTest, NoOverlapConfirmsPreviousState) {
  LevenshteinTextMerger merger;

  TextMerger::MergeResult res1;
  EXPECT_TRUE(merger.Merge(ToWords({"apple", "banana"}), &res1).ok());

  // Chunk 2 has no overlap
  TextMerger::MergeResult result;
  ASSERT_TRUE(merger.Merge(ToWords({"cat", "dog"}), &result).ok());
  EXPECT_EQ(result.confirmed_text, "apple banana");
  EXPECT_EQ(result.unconfirmed_text, "cat dog");
}

}  // namespace
}  // namespace litert_lm::omni::asr
