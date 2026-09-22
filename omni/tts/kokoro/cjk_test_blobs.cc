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

#include "omni/tts/kokoro/cjk_test_blobs.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace litert::omni::tts::kokoro {
void AppendUint32(uint32_t value, std::string* out) {
  out->push_back(static_cast<char>(value & 0xFF));
  out->push_back(static_cast<char>((value >> 8) & 0xFF));
  out->push_back(static_cast<char>((value >> 16) & 0xFF));
  out->push_back(static_cast<char>((value >> 24) & 0xFF));
}

void AppendUint16(uint16_t value, std::string* out) {
  out->push_back(static_cast<char>(value & 0xFF));
  out->push_back(static_cast<char>((value >> 8) & 0xFF));
}

std::string BuildCjkBlob(
    const std::vector<std::pair<std::string, std::string>>& sections) {
  std::string header = "LTTSCJK1";
  AppendUint32(1, &header);
  AppendUint32(sections.size(), &header);

  const int table_size = 16 * sections.size();
  int body_start = header.size() + table_size;
  body_start += (8 - body_start % 8) % 8;

  std::string table;
  std::string body;
  int cursor = body_start;
  for (const auto& [name, payload] : sections) {
    std::string padded_name = name;
    padded_name.resize(8, '\0');
    table += padded_name;
    AppendUint32(cursor, &table);
    AppendUint32(payload.size(), &table);
    body += payload;
    const int padding = (8 - payload.size() % 8) % 8;
    body.append(padding, '\0');
    cursor += payload.size() + padding;
  }
  std::string blob = header + table;
  blob.append(body_start - blob.size(), '\0');
  return blob + body;
}

}  // namespace litert::omni::tts::kokoro
