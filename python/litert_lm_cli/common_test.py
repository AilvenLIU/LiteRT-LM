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

from absl.testing import absltest

from litert_lm_cli import common


class CommonTest(absltest.TestCase):

  def test_size_string_from_bytes(self):
    # pylint: disable=protected-access
    self.assertEqual(common._size_string_from_bytes(0), "0B")
    self.assertEqual(common._size_string_from_bytes(512), "512B")
    self.assertEqual(common._size_string_from_bytes(1023), "1023B")
    self.assertEqual(common._size_string_from_bytes(1024), "1.0KiB")
    self.assertEqual(common._size_string_from_bytes(1536), "1.5KiB")
    self.assertEqual(common._size_string_from_bytes(1048576), "1.0MiB")
    self.assertEqual(common._size_string_from_bytes(1073741824), "1.0GiB")
    self.assertEqual(common._size_string_from_bytes(1099511627776), "1.0TiB")
    self.assertEqual(common._size_string_from_bytes(1125899906842624), "1.0PiB")
    # Test large number that exceeds PiB
    self.assertEqual(
        common._size_string_from_bytes(1125899906842624 * 1024), "1024.0PiB"
    )

  def test_download_size_suffix(self):
    self.assertEqual(common.download_size_suffix(None), "")
    self.assertEqual(common.download_size_suffix(1024), " (1.0KiB)")

  def test_format_download_progress(self):
    # With total size
    self.assertEqual(common.format_download_progress(50, 100), "50%")
    self.assertEqual(common.format_download_progress(0, 100), "0%")

    self.assertEqual(common.format_download_progress(500, None), "0.5 KiB")
    self.assertEqual(common.format_download_progress(1024, None), "1.0 KiB")
    self.assertEqual(
        common.format_download_progress(1048576, None), "1024.0 KiB"
    )
    self.assertEqual(common.format_download_progress(1048577, None), "1.0 MiB")

  def test_parse_total_size(self):
    self.assertEqual(common.parse_total_size("12345"), 12345)
    self.assertIsNone(common.parse_total_size(None))
    self.assertIsNone(common.parse_total_size("invalid"))


if __name__ == "__main__":
  absltest.main()
