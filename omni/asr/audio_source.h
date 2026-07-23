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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_

#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert_lm::omni::asr {

// Abstract interface for providing raw PCM audio data.
class AudioSource {
 public:
  virtual ~AudioSource() = default;

  // Resets internal state for a new audio stream.
  virtual void Reset() = 0;

  // Reads the next chunk of raw PCM float samples.
  // Returns absl::OutOfRangeError when end of stream is reached.
  virtual absl::StatusOr<std::vector<float>> GetNextChunk() = 0;

  // Reads the next chunk of raw PCM float samples asynchronously.
  // Note: Callbacks can be invoked on any thread, and may be called
  // synchronously before returning on the same thread especially on error.
  // It is the caller's responsibility to synchronize resources properly.
  virtual void GetNextChunkAsync(
      absl::AnyInvocable<void(absl::StatusOr<std::vector<float>>) &&>
          callback) = 0;

  // Returns the audio sampling rate in Hertz (e.g. 16000).
  virtual int GetSampleRateHz() const = 0;

  // Returns the number of audio channels (e.g. 1 for mono).
  virtual int GetNumChannels() const = 0;
};

}  // namespace litert_lm::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_
