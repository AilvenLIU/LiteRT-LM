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

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert_lm::omni::asr {
namespace {

TEST(TokenMergerTest, InitialChunkReturnsUnconfirmedOnly) {
  TokenMerger merger;
  std::vector<std::string> curr_chunk = {"hello", "world"};

  MergeResult result = merger.Merge(curr_chunk);

  EXPECT_EQ(result.confirmed_text, "");
  EXPECT_EQ(result.unconfirmed_text, "hello world");
}

TEST(TokenMergerTest, SequentialMergeFlow) {
  TokenMerger merger;

  // Chunk 1: "hello world this is"
  MergeResult res1 = merger.Merge({"hello", "world", "this", "is"});
  EXPECT_EQ(res1.confirmed_text, "");
  EXPECT_EQ(res1.unconfirmed_text, "hello world this is");

  // Chunk 2: overlaps at "this is", adds "a test"
  MergeResult res2 = merger.Merge({"this", "is", "a", "test"});
  EXPECT_EQ(res2.confirmed_text, "hello world");
  EXPECT_EQ(res2.unconfirmed_text, "this is a test");

  // Chunk 3: overlaps at "a test", adds "of streaming"
  MergeResult res3 = merger.Merge({"a", "test", "of", "streaming"});
  EXPECT_EQ(res3.confirmed_text, "this is");
  EXPECT_EQ(res3.unconfirmed_text, "a test of streaming");

  // End of stream flush
  MergeResult res_flush = merger.Flush();
  EXPECT_EQ(res_flush.confirmed_text, "a test of streaming");
  EXPECT_EQ(res_flush.unconfirmed_text, "");
}

TEST(TokenMergerTest, ResetClearsState) {
  TokenMerger merger;

  merger.Merge({"hello", "world"});
  EXPECT_FALSE(merger.unconfirmed_words().empty());

  merger.Reset();
  EXPECT_TRUE(merger.unconfirmed_words().empty());

  // First merge after reset behaves as initial chunk
  MergeResult result = merger.Merge({"new", "stream"});
  EXPECT_EQ(result.confirmed_text, "");
  EXPECT_EQ(result.unconfirmed_text, "new stream");
}

TEST(TokenMergerTest, NoOverlapConfirmsPreviousState) {
  TokenMerger merger;

  merger.Merge({"apple", "banana"});

  // Chunk 2 has no overlap
  MergeResult result = merger.Merge({"cat", "dog"});
  EXPECT_EQ(result.confirmed_text, "apple banana");
  EXPECT_EQ(result.unconfirmed_text, "cat dog");
}

}  // namespace
}  // namespace litert_lm::omni::asr
