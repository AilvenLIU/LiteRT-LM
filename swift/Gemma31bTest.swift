// Copyright 2026 Google LLC
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

import LiteRTLM
import XCTest

class Gemma31bTest: XCTestCase {

  private func createInitializedEngine() async throws -> Engine {
    let documentsUrl = try XCTUnwrap(
      FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first)

    let env = ProcessInfo.processInfo.environment
    let modelFileName = env["MODEL_FILENAME"] ?? "model.litertlm"
    let modelPath = documentsUrl.appendingPathComponent(modelFileName).path

    let backendStr = env["BACKEND"] ?? "gpu"
    let backend: LiteRTLM.Backend = (backendStr == "cpu") ? .cpu() : .gpu

    let engineConfig = try EngineConfig(
      modelPath: modelPath,
      backend: backend,
      maxNumTokens: 4096,
      cacheDir: NSTemporaryDirectory()
    )

    let engine = Engine(engineConfig: engineConfig)
    try await engine.initialize()
    let initialized = await engine.isInitialized()
    XCTAssertTrue(initialized)
    return engine
  }

  func testTextGenerationSucceeds() async throws {
    let engine = try await createInitializedEngine()
    let conversation = try await engine.createConversation()
    XCTAssertTrue(conversation.isAlive)

    // Turn 1
    let response1 = try await conversation.sendMessage(Message("What is the capital of France?"))
    XCTAssertEqual(response1.contents.count, 1)
    if case .text(let text) = response1.contents[0] {
      XCTAssertTrue(text.contains("Paris"), "Expected Paris in: \(text)")
    } else {
      XCTFail("Response should be text")
    }

    // Turn 2
    let response2 = try await conversation.sendMessage(Message("How about USA?"))
    XCTAssertEqual(response2.contents.count, 1)
    if case .text(let text) = response2.contents[0] {
      XCTAssertTrue(text.contains("Washington"), "Expected Washington in: \(text)")
    } else {
      XCTFail("Response should be text")
    }
  }

  func testResponseWithJsonSetBySystemInstructionSucceeds() async throws {
    let engine = try await createInitializedEngine()
    let conversation = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: Message("Output in JSON Object format.")
      )
    )
    XCTAssertTrue(conversation.isAlive)

    let response1 = try await conversation.sendMessage(Message("What is the capital of France?"))
    XCTAssertEqual(response1.contents.count, 1)
    if case .text(let text) = response1.contents[0] {
      XCTAssertTrue(isValidJsonObject(text), "Response should be valid JSON: \(text)")
      XCTAssertTrue(text.contains("Paris"), "Expected Paris in: \(text)")
    } else {
      XCTFail("Response should be text")
    }

    let response2 = try await conversation.sendMessage(Message("How about Japan?"))
    XCTAssertEqual(response2.contents.count, 1)
    if case .text(let text) = response2.contents[0] {
      XCTAssertTrue(isValidJsonObject(text), "Response should be valid JSON: \(text)")
      XCTAssertTrue(text.contains("Tokyo"), "Expected Tokyo in: \(text)")
    } else {
      XCTFail("Response should be text")
    }
  }

  func testResponseWithEmojiSucceeds() async throws {
    let engine = try await createInitializedEngine()
    let conversation = try await engine.createConversation()
    XCTAssertTrue(conversation.isAlive)

    let response = try await conversation.sendMessage(Message("What is the emoji of strawberry?"))
    XCTAssertEqual(response.contents.count, 1)
    if case .text(let text) = response.contents[0] {
      XCTAssertTrue(text.contains("🍓"), "Expected 🍓 in: \(text)")
    } else {
      XCTFail("Response should be text")
    }
  }

  func testRecreateConversationAndSendMessageSucceeds() async throws {
    let engine = try await createInitializedEngine()

    let conversation1 = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: Message("Reply in exactly one word.")
      )
    )
    XCTAssertTrue(conversation1.isAlive)

    let conversation2 = try await engine.createConversation(
      with: ConversationConfig(
        systemMessage: Message("Reply in exactly one word.")
      )
    )
    XCTAssertTrue(conversation2.isAlive)

    try await verifyConversationWorks(conversation1)
    try await verifyConversationWorks(conversation2)
  }

  private func verifyConversationWorks(_ conversation: Conversation) async throws {
    let response = try await conversation.sendMessage(Message("What is the capital of France?"))
    XCTAssertEqual(response.contents.count, 1)
    if case .text(let text) = response.contents[0] {
      let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
      XCTAssertTrue(trimmed == "Paris" || trimmed == "Paris.", "Expected Paris, got: \(trimmed)")
    } else {
      XCTFail("Response should be text")
    }
  }

  private func isValidJsonObject(_ cleanText: String) -> Bool {
    guard
      let data = cleanText.trimmingCharacters(in: .whitespacesAndNewlines)
        .deletingPrefix("```json")
        .deletingSuffix("```")
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .data(using: .utf8)
    else {
      return false
    }
    return (try? JSONSerialization.jsonObject(with: data, options: [])) is [String: Any]
  }
}

extension String {
  func deletingPrefix(_ prefix: String) -> String {
    guard self.hasPrefix(prefix) else { return self }
    return String(self.dropFirst(prefix.count))
  }
  func deletingSuffix(_ suffix: String) -> String {
    guard self.hasSuffix(suffix) else { return self }
    return String(self.dropLast(suffix.count))
  }
}
