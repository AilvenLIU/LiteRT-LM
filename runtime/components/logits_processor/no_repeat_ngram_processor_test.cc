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

#include "runtime/components/logits_processor/no_repeat_ngram_processor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;
using ::testing::status::StatusIs;

enum class InputType {
  kFloat,
  kHalf,
  kTensorBufferFloat,
  kTensorBufferHalf,
};

struct TestParam {
  int ngram_size;
  int window_size;
  InputType input_type;
};

std::vector<TestParam> GetTestParams() {
  std::vector<TestParam> params;
  for (int ngram_size : {1, 2, 3}) {
    for (int window_size : {0, ngram_size, ngram_size + 1, ngram_size + 2}) {
      for (InputType input_type :
           {InputType::kFloat, InputType::kHalf, InputType::kTensorBufferFloat,
            InputType::kTensorBufferHalf}) {
        params.push_back({ngram_size, window_size, input_type});
      }
    }
  }
  return params;
}

class NoRepeatNgramProcessorParamTest : public TestWithParam<TestParam> {
 protected:
  absl::Status UpdateProcessorState(NoRepeatNgramProcessor& processor,
                                    absl::Span<const int> tokens) {
    std::vector<int> tokens_copy(tokens.begin(), tokens.end());
    if (GetParam().input_type == InputType::kTensorBufferFloat ||
        GetParam().input_type == InputType::kTensorBufferHalf) {
      auto tb_tokens = CopyToTensorBuffer<int>(
          absl::MakeSpan(tokens_copy),
          /*dims=*/{static_cast<::litert::Layout::Dim>(tokens.size()), 1});
      if (!tb_tokens.HasValue()) {
        return absl::InternalError("Failed to copy tokens to TensorBuffer");
      }
      return processor.UpdateState(*tb_tokens);
    } else {
      return processor.UpdateState(absl::MakeSpan(tokens_copy));
    }
  }

  std::vector<float> ProcessProcessorLogits(
      NoRepeatNgramProcessor& processor, const std::vector<float>& logits,
      absl::Span<const ::litert::Layout::Dim> dims) {
    switch (GetParam().input_type) {
      case InputType::kFloat: {
        std::vector<float> logits_copy = logits;
        EXPECT_OK(processor.ProcessLogits(absl::MakeSpan(logits_copy), dims));
        return logits_copy;
      }
      case InputType::kHalf: {
        std::vector<tflite::half> logits_half(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
          logits_half[i] = static_cast<tflite::half>(logits[i]);
        }
        EXPECT_OK(processor.ProcessLogits(absl::MakeSpan(logits_half), dims));
        std::vector<float> result(logits.size());
        for (size_t i = 0; i < result.size(); ++i) {
          result[i] = static_cast<float>(logits_half[i]);
        }
        return result;
      }
      case InputType::kTensorBufferFloat: {
        std::vector<float> logits_copy = logits;
        ::litert::Dimensions tb_dims(dims.begin(), dims.end());
        auto tb_logits = CopyToTensorBuffer<float>(absl::MakeSpan(logits_copy),
                                                   std::move(tb_dims));
        EXPECT_TRUE(tb_logits.HasValue());
        EXPECT_OK(processor.ProcessLogits(*tb_logits));
        auto modified = CopyFromTensorBuffer<float>(*tb_logits);
        EXPECT_TRUE(modified.HasValue());
        if (modified.HasValue()) {
          return *modified;
        }
        return logits;
      }
      case InputType::kTensorBufferHalf: {
        std::vector<tflite::half> logits_half(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
          logits_half[i] = static_cast<tflite::half>(logits[i]);
        }
        ::litert::Dimensions tb_dims(dims.begin(), dims.end());
        auto tb_logits = CopyToTensorBuffer<tflite::half>(
            absl::MakeSpan(logits_half), std::move(tb_dims));
        EXPECT_TRUE(tb_logits.HasValue());
        EXPECT_OK(processor.ProcessLogits(*tb_logits));
        auto modified = CopyFromTensorBuffer<tflite::half>(*tb_logits);
        EXPECT_TRUE(modified.HasValue());
        std::vector<float> result(logits.size());
        if (modified.HasValue()) {
          for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<float>((*modified)[i]);
          }
        }
        return result;
      }
    }
  }
};

TEST_P(NoRepeatNgramProcessorParamTest, BansSeenNgramsAccordingToWindowSize) {
  const int ngram_size = GetParam().ngram_size;
  const int window_size = GetParam().window_size;
  NoRepeatNgramConfig config(ngram_size, window_size);
  const int vocab_size = 10;
  NoRepeatNgramProcessor processor(/*batch_size=*/1, vocab_size, config);

  EXPECT_EQ(processor.GetTokenHistorySizeForTesting(0), window_size);
  EXPECT_EQ(processor.GetPrefixHashHistorySizeForTesting(0),
            ngram_size > 1 ? window_size : 0);

  std::vector<int> prefix;
  for (int i = 0; i < ngram_size - 1; ++i) {
    prefix.push_back(i);
  }
  const int target_token = ngram_size - 1;
  const int filler_token = ngram_size;

  for (int token_id : prefix) {
    EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(&token_id, 1)));
  }
  EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(&target_token, 1)));
  EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(&filler_token, 1)));
  for (int token_id : prefix) {
    EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(&token_id, 1)));
  }

  std::vector<float> initial_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    initial_logits[i] = 1.0f + 0.5f * i;
  }
  std::vector<::litert::Layout::Dim> dims = {1, 1, vocab_size};

  std::vector<float> processed_logits =
      ProcessProcessorLogits(processor, initial_logits, dims);

  const float expected_banned_val =
      (GetParam().input_type == InputType::kHalf ||
       GetParam().input_type == InputType::kTensorBufferHalf)
          ? static_cast<float>(tflite::half::min())
          : -std::numeric_limits<float>::infinity();

  // The target token_id should be banned if the window size is large enough to
  // contain the prefix pattern twice, which is 2 * (ngram_size - 1) + 2.
  const bool should_ban =
      window_size == 0 || (ngram_size == 1 && window_size >= ngram_size + 1) ||
      (ngram_size > 1 && window_size >= 2 * (ngram_size - 1) +
                                            /*target_token*/ 1 +
                                            /*filler_token*/ 1);

  if (should_ban) {
    EXPECT_FLOAT_EQ(processed_logits[target_token], expected_banned_val);
  } else {
    EXPECT_FLOAT_EQ(processed_logits[target_token],
                    initial_logits[target_token]);
  }

  for (int i = 0; i < vocab_size; ++i) {
    if (i == target_token) continue;
    if (ngram_size == 1 && i == filler_token) {
      EXPECT_FLOAT_EQ(processed_logits[i], expected_banned_val);
      continue;
    }
    EXPECT_FLOAT_EQ(processed_logits[i], initial_logits[i]);
  }

  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 0);
}

TEST_P(NoRepeatNgramProcessorParamTest, HandlesBatchedSequences) {
  const int ngram_size = GetParam().ngram_size;
  const int window_size = GetParam().window_size;
  NoRepeatNgramConfig config(ngram_size, window_size);
  const int vocab_size = 10;
  NoRepeatNgramProcessor processor(/*batch_size=*/2, vocab_size, config);

  std::vector<int> prefix;
  for (int i = 0; i < ngram_size - 1; ++i) {
    prefix.push_back(i);
  }
  const int target_token1 = ngram_size - 1;
  const int target_token2 = ngram_size;
  const int filler_token1 = ngram_size + 1;
  const int filler_token2 = ngram_size + 2;

  for (int token_id : prefix) {
    std::vector<int> batch_tokens = {token_id, token_id};
    EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(batch_tokens)));
  }
  std::vector<int> step1_tokens = {target_token1, target_token2};
  EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(step1_tokens)));
  std::vector<int> step2_tokens = {filler_token1, filler_token2};
  EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(step2_tokens)));

  for (int token_id : prefix) {
    std::vector<int> batch_tokens = {token_id, token_id};
    EXPECT_OK(UpdateProcessorState(processor, absl::MakeSpan(batch_tokens)));
  }

  std::vector<float> initial_logits(vocab_size * 2);
  for (int i = 0; i < vocab_size * 2; ++i) {
    initial_logits[i] = 1.0f + 0.5f * i;
  }
  std::vector<::litert::Layout::Dim> dims = {2, 1, vocab_size};

  std::vector<float> processed_logits =
      ProcessProcessorLogits(processor, initial_logits, dims);

  const float expected_banned_val =
      (GetParam().input_type == InputType::kHalf ||
       GetParam().input_type == InputType::kTensorBufferHalf)
          ? static_cast<float>(tflite::half::min())
          : -std::numeric_limits<float>::infinity();

  // The target token_id should be banned if the window size is large enough to
  // contain the prefix pattern twice, which is 2 * (ngram_size - 1) + 2.
  const bool should_ban =
      window_size == 0 || (ngram_size == 1 && window_size >= ngram_size + 1) ||
      (ngram_size > 1 && window_size >= 2 * (ngram_size - 1) +
                                            /*target_token*/ 1 +
                                            /*filler_token*/ 1);

  // For Seq 0: target_token1 was seen after prefix.
  if (should_ban) {
    EXPECT_FLOAT_EQ(processed_logits[target_token1], expected_banned_val);
  } else {
    EXPECT_FLOAT_EQ(processed_logits[target_token1],
                    initial_logits[target_token1]);
  }

  // For Seq 1: target_token2 was seen after prefix.
  if (should_ban) {
    EXPECT_FLOAT_EQ(processed_logits[vocab_size + target_token2],
                    expected_banned_val);
  } else {
    EXPECT_FLOAT_EQ(processed_logits[vocab_size + target_token2],
                    initial_logits[vocab_size + target_token2]);
  }

  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 0);
}

TEST_P(NoRepeatNgramProcessorParamTest, HandlesNotEnoughTokensGeneratedYet) {
  const int ngram_size = GetParam().ngram_size;
  const int window_size = GetParam().window_size;
  NoRepeatNgramConfig config(ngram_size, window_size);
  const int vocab_size = 10;
  NoRepeatNgramProcessor processor(/*batch_size=*/1, vocab_size, config);
  const int fixed_token_id = 0;

  // We should not ban any tokens until we have generated ngram_size tokens.
  // then we should ban the fixed_token_id as it is the ngram_size-th token.
  for (int i = 0; i < ngram_size; ++i) {
    EXPECT_OK(
        UpdateProcessorState(processor, absl::MakeSpan(&fixed_token_id, 1)));

    std::vector<float> initial_logits(vocab_size, 2.0f);
    std::vector<::litert::Layout::Dim> dims = {1, 1, vocab_size};

    std::vector<float> processed_logits =
        ProcessProcessorLogits(processor, initial_logits, dims);

    for (int token_id = 0; token_id < vocab_size; ++token_id) {
      if (i >= ngram_size - 1 && token_id == fixed_token_id) {
        const float expected_banned_val =
            (GetParam().input_type == InputType::kHalf ||
             GetParam().input_type == InputType::kTensorBufferHalf)
                ? static_cast<float>(tflite::half::min())
                : -std::numeric_limits<float>::infinity();
        EXPECT_FLOAT_EQ(processed_logits[token_id], expected_banned_val);
      } else {
        EXPECT_FLOAT_EQ(processed_logits[token_id], 2.0f);
      }
    }
  }

  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    NoRepeatNgramProcessorParamTest, NoRepeatNgramProcessorParamTest,
    ValuesIn(GetTestParams()), [](const TestParamInfo<TestParam>& info) {
      std::string input_type_str;
      switch (info.param.input_type) {
        case InputType::kFloat:
          input_type_str = "Float";
          break;
        case InputType::kHalf:
          input_type_str = "Half";
          break;
        case InputType::kTensorBufferFloat:
          input_type_str = "TensorBufferFloat";
          break;
        case InputType::kTensorBufferHalf:
          input_type_str = "TensorBufferHalf";
          break;
      }
      return "Ngram" + std::to_string(info.param.ngram_size) + "_Window" +
             std::to_string(info.param.window_size) + "_" + input_type_str;
    });

TEST(NoRepeatNgramProcessorTest, HandlesNoRepeatNgramSizeDisabled) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/0, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/1, /*vocab_size=*/3, config);

  EXPECT_EQ(processor.GetTokenHistorySizeForTesting(0), 0);
  EXPECT_EQ(processor.GetPrefixHashHistorySizeForTesting(0), 0);

  std::vector<int> tokens = {1, 2, 1};
  for (int token_id : tokens) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  std::vector<float> logits = {2.0f, 1.2f, -1.2f};
  std::vector<::litert::Layout::Dim> dims = {1, 1, 3};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  EXPECT_FLOAT_EQ(logits[0], 2.0f);
  EXPECT_FLOAT_EQ(logits[1], 1.2f);
  EXPECT_FLOAT_EQ(logits[2], -1.2f);

  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 0);
}

TEST(NoRepeatNgramProcessorTest, HandlesMultiplePrefixOccurrencesInWindow) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/6);
  NoRepeatNgramProcessor processor(/*batch_size=*/1, /*vocab_size=*/10, config);

  std::vector<int> tokens = {1, 2, 1, 2, 3, 1, 2};
  for (int token_id : tokens) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  std::vector<float> logits(10, 1.0f);
  std::vector<::litert::Layout::Dim> dims = {1, 1, 10};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  // (1) is no longer banned because the first (1, 2) has left the window.
  EXPECT_FLOAT_EQ(logits[1], 1.0f);
  // (3) MUST be banned because it was seen after the second (1, 2), which is
  // still in the window.
  EXPECT_EQ(logits[3], -std::numeric_limits<float>::infinity());

  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 0);
}

TEST(NoRepeatNgramProcessorTest, HandlesUnigramMultipleOccurrencesInWindow) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/1, /*window_size=*/3);
  NoRepeatNgramProcessor processor(/*batch_size=*/1, /*vocab_size=*/5, config);

  // Update the token history to [0, 1, 1].
  std::vector<int> tokens = {0, 1, 1};
  for (int token_id : tokens) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  std::vector<float> logits(5, 1.0f);
  std::vector<::litert::Layout::Dim> dims = {1, 1, 5};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

  // (0) is banned because it is the 1st token.
  EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
  // (1) is banned because it is the 2nd and 3rd token.
  EXPECT_EQ(logits[1], -std::numeric_limits<float>::infinity());
  // (2) is not banned because it has not been seen before.
  EXPECT_EQ(logits[2], 1.0f);

  // Update the token history to [1, 1, 2].
  int token_id = 2;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  std::fill(logits.begin(), logits.end(), 1.0f);
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  // (0) is no longer banned because it is out of the window.
  EXPECT_EQ(logits[0], 1.0f);
  // (1) is still banned because it is the 2nd and 3rd token.
  EXPECT_EQ(logits[1], -std::numeric_limits<float>::infinity());
  // (2) is banned because it is the 4th token.
  EXPECT_EQ(logits[2], -std::numeric_limits<float>::infinity());

  // Update the token history to [1, 2, 0].
  token_id = 0;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  std::fill(logits.begin(), logits.end(), 1.0f);
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  // (0) is banned because it is the 5th token.
  EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
  // (1) is still banned because it is the 3rd token.
  EXPECT_EQ(logits[1], -std::numeric_limits<float>::infinity());
  // (2) is still banned because it is the 4th token.
  EXPECT_EQ(logits[2], -std::numeric_limits<float>::infinity());

  // Update the token history to [2, 0, 0].
  token_id = 0;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  std::fill(logits.begin(), logits.end(), 1.0f);
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  // (0) is still banned because it is the 5th and 6th token.
  EXPECT_EQ(logits[0], -std::numeric_limits<float>::infinity());
  // (1) is no longer banned because it is out of the window.
  EXPECT_EQ(logits[1], 1.0f);
  // (2) is still banned because it is the 4th token.
  EXPECT_EQ(logits[2], -std::numeric_limits<float>::infinity());
}

TEST(NoRepeatNgramProcessorTest, ReturnsErrorForInvalidLogitsDims) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<float> logits_float(10, 1.0f);
  std::vector<tflite::half> logits_half(10, static_cast<tflite::half>(1.0f));

  std::vector<std::vector<::litert::Layout::Dim>> invalid_dims = {
      {2, 5},     // Invalid rank (2 dims instead of 3)
      {1, 1, 5},  // Invalid batch size (1 instead of 2)
      {2, 2, 5},  // Invalid sequence length (2 instead of 1)
      {2, 1, 4},  // Invalid vocab size (4 instead of 5)
  };

  for (const auto& dims : invalid_dims) {
    EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_float),
                                        absl::MakeSpan(dims)),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_half),
                                        absl::MakeSpan(dims)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(NoRepeatNgramProcessorTest, ReturnsErrorForInvalidLogitsSize) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<::litert::Layout::Dim> dims = {2, 1, 5};
  std::vector<float> logits(11, 1.0f);  // Size 11 instead of 10

  EXPECT_THAT(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Logits span size incorrectly mapped."));
}

TEST(NoRepeatNgramProcessorTest,
     ReturnsErrorForInvalidTensorBufferElementType) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  std::vector<::litert::Layout::Dim> tokens_dims = {2, 1};
  std::vector<::litert::Layout::Dim> logits_dims = {2, 1, 5};

  // UpdateState with float TensorBuffer should fail.
  std::vector<float> tokens_float(2, 1.0f);
  auto tb_tokens_float = CopyToTensorBuffer<float>(
      absl::MakeSpan(tokens_float),
      ::litert::Dimensions(tokens_dims.begin(), tokens_dims.end()));
  ASSERT_TRUE(tb_tokens_float.HasValue());
  EXPECT_FALSE(processor.UpdateState(*tb_tokens_float).ok());

  // ProcessLogits with Int32 TensorBuffer should fail.
  std::vector<int32_t> logits_int32(10, 1);
  auto tb_logits_int32 = CopyToTensorBuffer<int32_t>(
      absl::MakeSpan(logits_int32),
      ::litert::Dimensions(logits_dims.begin(), logits_dims.end()));
  ASSERT_TRUE(tb_logits_int32.HasValue());
  EXPECT_THAT(processor.ProcessLogits(*tb_logits_int32),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported logits tensor format element type."));
}

TEST(NoRepeatNgramProcessorTest, ReturnsErrorForInvalidUpdateState) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/2, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/2, /*vocab_size=*/5, config);

  // Wrong batch size (1 instead of 2)
  std::vector<int> tokens_wrong_size = {1};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_wrong_size)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "next_token_ids size must match batch_size"));

  // Invalid token ID (-1)
  std::vector<int> tokens_invalid_negative = {1, -1};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_invalid_negative)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "next_token_ids contains invalid token id"));

  // Invalid token ID (equal to vocab_size 5)
  std::vector<int> tokens_invalid_large = {1, 5};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_invalid_large)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "next_token_ids contains invalid token id"));
}

TEST(NoRepeatNgramProcessorTest,
     HandlesHashCollisionWithPrefixesMatchWithoutWindowSize) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/3, /*window_size=*/0);
  NoRepeatNgramProcessor processor(/*batch_size=*/1, /*vocab_size=*/10, config);

  // Generate prefix A: [0, 1] followed by token_id 3.
  std::vector<int> prefix_a = {0, 1};
  for (int token_id : prefix_a) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  // At this point, the prefix [0, 1] is formed and hashed. Get its hash.
  size_t hash_a = processor.GetLatestPrefixHashForTesting(/*batch_idx=*/0);

  // Generate prefix B: [1, 1].
  int token_id = 1;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Force hash collision: set prefix B's hash to hash A.
  processor.SetLatestPrefixHashForTesting(/*batch_idx=*/0, hash_a);

  // Now, Test if processor bans token 1 (the next token of prefix A).
  // It shouldn't, because PrefixesMatch will detect [1, 1] != [0, 1].
  std::vector<float> logits(10, 1.0f);
  std::vector<::litert::Layout::Dim> dims = {1, 1, 10};

  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 1);

  // The token_id 1 should not be banned! (banned is -inf)
  EXPECT_FLOAT_EQ(logits[1], 1.0f);
}

TEST(NoRepeatNgramProcessorTest,
     HandlesHashCollisionWithPrefixesMatchWithWindowSize) {
  NoRepeatNgramConfig config(/*no_repeat_ngram_size=*/4, /*window_size=*/4);
  NoRepeatNgramProcessor processor(/*batch_size=*/1, /*vocab_size=*/10, config);
  std::vector<float> logits(10, 1.0f);
  std::vector<::litert::Layout::Dim> dims = {1, 1, 10};
  int token_id = 0;

  // Case 1: no wrap-around for previous prefix and the current prefix.
  // Generate prefix A: [0, 0, 0] followed by token_id 1.
  std::vector<int> prefix_a = {0, 0, 0};
  token_id = 1;
  for (int token_id : prefix_a) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  // At this point, the prefix [0, 0, 0] is formed and hashed. Get its hash.
  size_t hash_a = processor.GetLatestPrefixHashForTesting(/*batch_idx=*/0);

  // Generate prefix B: [0, 0, 1].
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Force hash collision: set prefix B's hash to hash A.
  processor.SetLatestPrefixHashForTesting(/*batch_idx=*/0, hash_a);

  // Test if processor bans token 1 (the next token of prefix A).
  // It shouldn't, because PrefixesMatch will detect [0, 0, 1] != [0, 0, 0].
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 1);

  // The token_id 1 should not be banned! (banned is -inf)
  EXPECT_FLOAT_EQ(logits[token_id], 1.0f);

  // Current token history: [0, 0, 0, 1]
  // Current token history index: 3

  // Case 2: wrap-around for current prefix only.
  // Update a random token to index 0.
  // Current token history: [9, 0, 0, 0]
  // Current token history index: 1
  token_id = 9;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Generate prefix C: [1, 1, 1] followed by token_id 2.
  std::vector<int> prefix_c = {1, 1, 1};
  token_id = 2;
  for (int token_id : prefix_c) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  // Current token history: [9, 1, 1, 1]
  // Current token history index: 0
  // At this point, the prefix [1, 1, 1] is formed and hashed. Get its hash.
  size_t hash_c = processor.GetLatestPrefixHashForTesting(/*batch_idx=*/0);

  // Update state with token 2 and generate a new prefix D: [1, 1, 2].
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Force hash collision: set prefix D's hash to hash C.
  processor.SetLatestPrefixHashForTesting(/*batch_idx=*/0, hash_c);

  // Test if processor bans token 2 (the next token of prefix C).
  // It shouldn't, because PrefixesMatch will detect [1, 1, 2] != [1, 1, 1].
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 2);

  // The token_id 2 should not be banned! (banned is -inf)
  EXPECT_FLOAT_EQ(logits[token_id], 1.0f);

  // Current token history: [2, 1, 1, 1]
  // Current token history index: 1

  // Case 3: wrap-around for both previous prefix and the current prefix.
  // Update a random token to index 1.
  // Current token history: [2, 9, 1, 1]
  // Current token history index: 2
  token_id = 9;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Generate prefix E: [2, 2, 2] followed by token_id 3.
  std::vector<int> prefix_e = {2, 2, 2};
  token_id = 3;
  for (int token_id : prefix_e) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  // Current token history: [2, 9, 2, 2]
  // Current token history index: 1
  // At this point, the prefix [2, 2, 2] is formed and hashed. Get its hash.
  size_t hash_e = processor.GetLatestPrefixHashForTesting(/*batch_idx=*/0);

  // Update state with token 3 and generate a new prefix F: [2, 2, 3].
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Force hash collision: set prefix F's hash to hash E.
  processor.SetLatestPrefixHashForTesting(/*batch_idx=*/0, hash_e);

  // Test if processor bans token 3 (the next token of prefix E).
  // It shouldn't, because PrefixesMatch will detect [2, 2, 3] != [2, 2, 2].
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 3);

  // The token_id 3 should not be banned! (banned is -inf)
  EXPECT_FLOAT_EQ(logits[token_id], 1.0f);

  // Current token history: [2, 3, 2, 2]
  // Current token history index: 2

  // Case 4: wrap-around for the previous prefix only.
  // Update a random token to index 2.
  // Current token history: [2, 3, 9, 2]
  // Current token history index: 3
  token_id = 9;
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Generate prefix E: [3, 3, 3] followed by token_id 4.
  std::vector<int> prefix_g = {3, 3, 3};
  token_id = 4;
  for (int token_id : prefix_g) {
    EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));
  }

  // Current token history: [3, 3, 9, 3]
  // Current token history index: 2
  // At this point, the prefix [3, 3, 3] is formed and hashed. Get its hash.
  size_t hash_g = processor.GetLatestPrefixHashForTesting(/*batch_idx=*/0);

  // Update state with token 4 and generate a new prefix H: [3, 3, 4].
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(&token_id, 1)));

  // Force hash collision: set prefix H's hash to hash G.
  processor.SetLatestPrefixHashForTesting(/*batch_idx=*/0, hash_g);

  // Test if processor bans token 4 (the next token of prefix G).
  // It shouldn't, because PrefixesMatch will detect [3, 3, 4] != [3, 3, 3].
  EXPECT_OK(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));
  EXPECT_EQ(processor.GetPrefixHashCollisionCountForTesting(), 4);

  // The token_id 4 should not be banned! (banned is -inf)
  EXPECT_FLOAT_EQ(logits[token_id], 1.0f);
}

}  // namespace
}  // namespace litert::lm
