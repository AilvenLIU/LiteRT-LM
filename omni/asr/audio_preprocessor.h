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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert_lm::omni::asr {

// Abstract interface for preprocessing raw PCM audio into features.
class AudioPreprocessor {
 public:
  virtual ~AudioPreprocessor() = default;

  // Resets internal cached state for a new audio stream.
  virtual void Reset() = 0;

  // Preprocesses pcm_samples into output_mel_features without unnecessary copy.
  virtual absl::Status Preprocess(absl::Span<const float> pcm_samples,
                                  std::vector<float>* output_mel_features) = 0;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_
