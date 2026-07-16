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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "c/engine.h"
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/token.pb.h"

struct LiteRtLmInputData {
  explicit LiteRtLmInputData(litert::lm::InputData d) : data(std::move(d)) {}
  litert::lm::InputData data;
};

struct LiteRtLmSamplerParams {
  LiteRtLmSamplerType type;
  int32_t top_k;
  float top_p;
  float temperature;
  int32_t seed;
};

struct LiteRtLmStreamChunk {
  const char* text = nullptr;
  bool is_final = false;
  const char* error_msg = nullptr;
};

struct LiteRtLmRepetitionPenaltyConfig {
  litert::lm::RepetitionPenaltyConfig repetition_penalty_config;
};

struct LiteRtLmNoRepeatNgramConfig {
  litert::lm::NoRepeatNgramConfig no_repeat_ngram_config;
};

struct LiteRtLmSuppressTokensConfig {
  litert::lm::SuppressTokensConfig suppress_tokens_config;
};

struct LiteRtLmConversationOptionalArgs {
  std::optional<litert::lm::RepetitionPenaltyConfig> repetition_penalty_config;
  std::optional<litert::lm::NoRepeatNgramConfig> no_repeat_ngram_config;
  std::optional<litert::lm::SuppressTokensConfig> suppress_tokens_config;
  std::optional<int> visual_token_budget;
  std::optional<int> max_output_tokens;
  std::optional<litert::lm::ThinkingConfig> thinking_config;
  LiteRtLmConstraintType constraint_type = kLiteRtLmConstraintTypeNone;
  std::string constraint_string;
};

struct LiteRtLmEngineSettings {
  std::unique_ptr<litert::lm::EngineSettings> settings;
};

struct LiteRtLmEngine {
  std::unique_ptr<litert::lm::Engine> engine;
};

struct LiteRtLmSession {
  std::unique_ptr<litert::lm::Engine::Session> session;
};

struct LiteRtLmResponses {
  litert::lm::Responses responses;
};

struct LiteRtLmBenchmarkInfo {
  litert::lm::BenchmarkInfo benchmark_info;
};

struct LiteRtLmConversation {
  std::unique_ptr<litert::lm::Conversation> conversation;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_message_to_string`. This ties the lifetime
  // of the returned `const char*` to the `LiteRtLmConversation` object,
  // ensuring memory safety for the C API caller without requiring explicit
  // per-call deallocation.
  std::string last_rendered_message;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_preface_to_string`.
  std::string last_rendered_preface;
};

struct LiteRtLmJsonResponse {
  std::string json_string;
};

// TODO: b/483172229 - Migrate to use SessionConfig instead of unique_ptr to
// SessionConfig for consistency and efficiency.
struct LiteRtLmSessionConfig {
  std::unique_ptr<litert::lm::SessionConfig> config;
};

struct LiteRtLmConversationConfig {
  std::optional<litert::lm::SessionConfig> session_config;
  std::string system_message_json;
  std::string tools_json;
  std::string messages_json;
  std::string extra_context_json;
  std::string prompt_template;
  bool enable_constrained_decoding = false;
  bool filter_channel_content_from_kv_cache = false;
  bool stream_tool_calls = false;
  std::string stream_tool_calls_channel_name = "tool_call";
  std::optional<litert::lm::ThinkingConfig> thinking_config;
  std::optional<LiteRtLmConstraintProviderType> constraint_provider_type;
};

struct LiteRtLmDetokenizeResult {
  std::string text;
};

struct LiteRtLmTokenizeResult {
  std::vector<int> tokens;
};

struct LiteRtLmTokenUnion {
  litert::lm::proto::TokenUnion token_union;
};

struct LiteRtLmTokenUnions {
  std::vector<litert::lm::proto::TokenUnion> tokens;
};

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_INTERNAL_H_
