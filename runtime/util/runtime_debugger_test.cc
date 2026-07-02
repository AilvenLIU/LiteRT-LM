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

#include "runtime/util/runtime_debugger.h"

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

TEST(RuntimeDebuggerTest, ConstructorStripsTrailingSlashes) {
  RuntimeDebugger debugger("/tmp/test_dir///");
}

TEST(RuntimeDebuggerTest, CreateReusesPreferredCacheDir) {
  std::string test_dir = testing::TempDir();
  ::unsetenv("LITERT_LM_DEBUG_DUMP_DIR");
  auto debugger = RuntimeDebugger::Create(test_dir);
  ASSERT_NE(debugger, nullptr);
  EXPECT_EQ(debugger->dump_dir(), test_dir);
}

TEST(RuntimeDebuggerTest, CreateReturnsNullptrForInMemoryOrNoCacheTokens) {
  auto debugger = RuntimeDebugger::Create(":memory");
  EXPECT_EQ(debugger, nullptr);
}

TEST(RuntimeDebuggerTest, ObserveTokensDumpsToDisk) {
  std::string test_dir = testing::TempDir();
  RuntimeDebugger debugger(test_dir);

  Responses responses(TaskState::kDone, {"test"}, {1.0f}, {2}, {{123, 456}});
  debugger.ObserveTokens(/*session_id=*/0, responses);

  std::string dumped_path = absl::StrCat(test_dir, "/generated_tokens.jsonl");
  std::ifstream infile(dumped_path);
  ASSERT_TRUE(infile.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(infile, line));
  EXPECT_THAT(line, testing::HasSubstr("\"token_ids\":[[123,456]]"));
  EXPECT_THAT(line, testing::HasSubstr("\"texts\":[\"test\"]"));
  EXPECT_THAT(line, testing::HasSubstr("\"scores\":[1.0]"));
}

TEST(RuntimeDebuggerTest, CreatePostGraphRunCallbackReturnsCallable) {
  RuntimeDebugger debugger("/tmp/dump");
  auto callback = debugger.CreatePostGraphRunCallback(nullptr);
  EXPECT_TRUE(callback != nullptr);
}

}  // namespace
}  // namespace litert::lm
