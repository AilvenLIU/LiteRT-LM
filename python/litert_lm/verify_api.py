# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone structural verification suite for pre-built LiteRT-LM PyPI wheels."""

import pathlib
import litert_lm


def verify_structural_smoke_suite(model_path: pathlib.Path):
  """Executes lightweight structural smoke tests proving core CPU engine stability."""
  print("------------------------------------------------")
  print(f"🧪 Running Structural Smoke Suite using {model_path}")
  print("------------------------------------------------")

  with (
      litert_lm.Engine(str(model_path), max_num_tokens=10) as engine,
      engine.create_conversation() as conversation,
  ):
    print("🧪 Sending test prompt...")
    message = conversation.send_message("Hello!")
    output_text = message["content"][0]["text"]
    print(f"   Engine Response: {output_text.strip()}")
    assert (
        isinstance(message, dict) and "content" in message
    ), "API Contract Breach: Response format invalid"

  print("🎉 Active Structural Verification completed flawlessly!")


def main():
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.INFO)

  # Robustly discover SDK Root
  sdk_root = pathlib.Path(__file__).resolve().parent.parent
  target_model = sdk_root / "runtime" / "testdata" / "test_lm.litertlm"

  if not target_model.exists():
    raise FileNotFoundError(
        "❌ CRITICAL ERROR: Verification failed to find target model at"
        f" {target_model}"
    )

  verify_structural_smoke_suite(target_model)


if __name__ == "__main__":
  main()
