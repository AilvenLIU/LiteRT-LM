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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/detokenizer.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/speech_decoder.h"
#include "omni/asr/text_merger.h"

namespace litert_lm::omni::asr {
namespace {

// Dummy AudioSource returning pre-configured PCM audio chunks.
class DummyAudioSource : public AudioSource {
 public:
  explicit DummyAudioSource(std::vector<std::vector<float>> chunks)
      : chunks_(std::move(chunks)) {}

  void Reset() override {}

  absl::StatusOr<std::vector<float>> GetNextChunk() override {
    if (chunk_index_ >= chunks_.size()) {
      return absl::OutOfRangeError("End of audio stream reached.");
    }
    return chunks_[chunk_index_++];
  }

  void GetNextChunkAsync(
      absl::AnyInvocable<void(absl::StatusOr<std::vector<float>>) &&> callback)
      override {
    auto res = GetNextChunk();
    std::move(callback)(res);
  }

  int GetSampleRateHz() const override { return 16000; }
  int GetNumChannels() const override { return 1; }

 private:
  std::vector<std::vector<float>> chunks_;
  size_t chunk_index_ = 0;
};

// Dummy AudioPreprocessor passing through dummy mel feature values.
class DummyAudioPreprocessor : public AudioPreprocessor {
 public:
  void Reset() override {}

  absl::StatusOr<std::vector<float>> Preprocess(
      absl::Span<const float> pcm_samples) override {
    return std::vector<float>(pcm_samples.begin(), pcm_samples.end());
  }

  void PreprocessAsync(
      absl::Span<const float> pcm_samples,
      absl::AnyInvocable<void(absl::StatusOr<std::vector<float>>) &&> callback)
      override {
    auto res = Preprocess(pcm_samples);
    std::move(callback)(res);
  }
};

// Dummy SpeechDecoder returning dummy DecodedToken IDs.
class DummySpeechDecoder : public SpeechDecoder {
 public:
  void Reset() override {}

  absl::StatusOr<std::vector<SpeechDecoder::DecodedToken>> Decode(
      absl::Span<const float> mel_features) override {
    std::vector<SpeechDecoder::DecodedToken> tokens;
    tokens.reserve(mel_features.size());
    for (size_t i = 0; i < mel_features.size(); ++i) {
      tokens.push_back({static_cast<int>(mel_features[i]), 100});
    }
    return tokens;
  }

  void DecodeAsync(
      absl::Span<const float> mel_features,
      absl::AnyInvocable<
          void(absl::StatusOr<std::vector<SpeechDecoder::DecodedToken>>) &&>
          callback) override {
    auto res = Decode(mel_features);
    std::move(callback)(res);
  }
};

// Dummy Detokenizer mapping token IDs to string words with std::optional
// timestamp.
class DummyDetokenizer : public Detokenizer {
 public:
  void Reset() override {}

  absl::StatusOr<std::vector<Detokenizer::Word>> Detokenize(
      absl::Span<const SpeechDecoder::DecodedToken> tokens) override {
    std::vector<Detokenizer::Word> words;
    words.reserve(tokens.size());
    for (const auto& tok : tokens) {
      words.push_back(
          {"word_" + std::to_string(tok.token_id), tok.timestamp_ms});
    }
    return words;
  }
};

TEST(AsrSessionTest, FullSessionEndToEndFlow) {
  // Setup dummy audio chunks representing token IDs 1, 2, 3
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
      {2.0f, 3.0f},
  };

  AsrSession::Components components;
  components.audio_source = std::make_unique<DummyAudioSource>(chunks);
  components.preprocessor = std::make_unique<DummyAudioPreprocessor>();
  components.speech_decoder = std::make_unique<DummySpeechDecoder>();
  components.detokenizer = std::make_unique<DummyDetokenizer>();
  components.text_merger = std::make_unique<LevenshteinTextMerger>();

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_TRUE(session_status.ok());
  auto session = std::move(*session_status);

  // Process Chunk 1: "word_1 word_2"
  auto res1 = session->ProcessNextChunk();
  ASSERT_TRUE(res1.ok());
  EXPECT_EQ(res1->confirmed_text, "");
  EXPECT_EQ(res1->unconfirmed_text, "word_1 word_2");

  // Process Chunk 2: "word_2 word_3" (overlaps at word_2)
  auto res2 = session->ProcessNextChunk();
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->confirmed_text, "word_1");
  EXPECT_EQ(res2->unconfirmed_text, "word_2 word_3");

  // Stream End returns OutOfRange error
  auto res3 = session->ProcessNextChunk();
  EXPECT_TRUE(absl::IsOutOfRange(res3.status()));

  // Flush remaining
  auto res_flush = session->Flush();
  ASSERT_TRUE(res_flush.ok());
  EXPECT_EQ(res_flush->confirmed_text, "word_2 word_3");
  EXPECT_EQ(res_flush->unconfirmed_text, "");
}

TEST(AsrSessionTest, ProcessNextChunkAsyncFlow) {
  std::vector<std::vector<float>> chunks = {
      {1.0f, 2.0f},
      {2.0f, 3.0f},
  };

  AsrSession::Components components;
  components.audio_source = std::make_unique<DummyAudioSource>(chunks);
  components.preprocessor = std::make_unique<DummyAudioPreprocessor>();
  components.speech_decoder = std::make_unique<DummySpeechDecoder>();
  components.detokenizer = std::make_unique<DummyDetokenizer>();
  components.text_merger = std::make_unique<LevenshteinTextMerger>();

  auto session_status = AsrSession::Create(std::move(components));
  ASSERT_TRUE(session_status.ok());
  auto session = std::move(*session_status);

  // Async Chunk 1
  bool called1 = false;
  session->ProcessNextChunkAsync([&called1](auto res) {
    called1 = true;
    ASSERT_TRUE(res.ok());
    EXPECT_EQ(res->confirmed_text, "");
    EXPECT_EQ(res->unconfirmed_text, "word_1 word_2");
  });
  EXPECT_TRUE(called1);

  // Async Chunk 2
  bool called2 = false;
  session->ProcessNextChunkAsync([&called2](auto res) {
    called2 = true;
    ASSERT_TRUE(res.ok());
    EXPECT_EQ(res->confirmed_text, "word_1");
    EXPECT_EQ(res->unconfirmed_text, "word_2 word_3");
  });
  EXPECT_TRUE(called2);

  // Async Chunk 3 (End of stream)
  bool called3 = false;
  session->ProcessNextChunkAsync([&called3](auto res) {
    called3 = true;
    EXPECT_TRUE(absl::IsOutOfRange(res.status()));
  });
  EXPECT_TRUE(called3);
}

TEST(AsrSessionTest, FailsWhenMissingComponent) {
  AsrSession::Components components;
  components.audio_source =
      std::make_unique<DummyAudioSource>(std::vector<std::vector<float>>{});
  // Intentionally leave preprocessor null

  auto session_status = AsrSession::Create(std::move(components));
  EXPECT_FALSE(session_status.ok());
}

}  // namespace
}  // namespace litert_lm::omni::asr
