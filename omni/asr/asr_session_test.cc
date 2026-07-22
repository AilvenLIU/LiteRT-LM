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
#include "absl/status/status.h"  // from @com_google_absl
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

  absl::Status GetNextChunk(std::vector<float>* output_pcm) override {
    if (chunk_index_ >= chunks_.size()) {
      return absl::OutOfRangeError("End of audio stream reached.");
    }
    *output_pcm = chunks_[chunk_index_++];
    return absl::OkStatus();
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

  absl::Status Preprocess(absl::Span<const float> pcm_samples,
                          std::vector<float>* output_mel_features) override {
    output_mel_features->assign(pcm_samples.begin(), pcm_samples.end());
    return absl::OkStatus();
  }
};

// Dummy SpeechDecoder returning dummy DecodedToken IDs.
class DummySpeechDecoder : public SpeechDecoder {
 public:
  void Reset() override {}

  absl::Status Decode(
      absl::Span<const float> mel_features,
      std::vector<SpeechDecoder::DecodedToken>* output_tokens) override {
    output_tokens->clear();
    for (size_t i = 0; i < mel_features.size(); ++i) {
      output_tokens->push_back({static_cast<int>(mel_features[i]), 100});
    }
    return absl::OkStatus();
  }
};

// Dummy Detokenizer mapping token IDs to string words.
class DummyDetokenizer : public Detokenizer {
 public:
  absl::Status Detokenize(
      absl::Span<const SpeechDecoder::DecodedToken> tokens,
      std::vector<Detokenizer::Word>* output_words) override {
    output_words->clear();
    for (const auto& tok : tokens) {
      output_words->push_back(
          {"word_" + std::to_string(tok.token_id), tok.timestamp_ms});
    }
    return absl::OkStatus();
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
  TextMerger::MergeResult res1;
  auto status1 = session->ProcessNextChunk(&res1);
  ASSERT_TRUE(status1.ok());
  EXPECT_EQ(res1.confirmed_text, "");
  EXPECT_EQ(res1.unconfirmed_text, "word_1 word_2");

  // Process Chunk 2: "word_2 word_3" (overlaps at word_2)
  TextMerger::MergeResult res2;
  auto status2 = session->ProcessNextChunk(&res2);
  ASSERT_TRUE(status2.ok());
  EXPECT_EQ(res2.confirmed_text, "word_1");
  EXPECT_EQ(res2.unconfirmed_text, "word_2 word_3");

  // Stream End returns OutOfRange error
  TextMerger::MergeResult res3;
  auto status3 = session->ProcessNextChunk(&res3);
  EXPECT_TRUE(absl::IsOutOfRange(status3));

  // Flush remaining
  TextMerger::MergeResult res_flush;
  auto flush_status = session->Flush(&res_flush);
  ASSERT_TRUE(flush_status.ok());
  EXPECT_EQ(res_flush.confirmed_text, "word_2 word_3");
  EXPECT_EQ(res_flush.unconfirmed_text, "");
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
