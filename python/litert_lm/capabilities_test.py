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
"""Tests for model capabilities extraction API."""

import pathlib

from absl import flags
from absl.testing import absltest

import litert_lm

FLAGS = flags.FLAGS


class CapabilitiesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = str(
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

  def test_capabilities_load(self):
    capabilities = litert_lm.Capabilities(self.model_path)

    # Check simple capability flags (expect False for the legacy test model)
    self.assertFalse(capabilities.supports_thinking())
    self.assertFalse(capabilities.supports_function_calling())
    self.assertFalse(capabilities.has_speculative_decoding_support())

    # Modalities
    self.assertTrue(capabilities.input_modalities.text)
    self.assertFalse(capabilities.input_modalities.vision)
    self.assertFalse(capabilities.input_modalities.audio)
    self.assertFalse(capabilities.input_modalities.video)

    # Sampler default parameters (from test model config)
    sampler_params = capabilities.default_sampler_params
    self.assertEqual(sampler_params.type, litert_lm.SamplerType.TOP_P)
    self.assertEqual(sampler_params.temperature, 0.0)
    self.assertEqual(sampler_params.top_k, 1)
    self.assertAlmostEqual(sampler_params.top_p, 0.7)

  def test_capabilities_non_existent_file(self):
    with self.assertRaises(FileNotFoundError):
      litert_lm.Capabilities("/non/existent/path")


if __name__ == "__main__":
  absltest.main()
