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

#include "runtime/util/safetensors_util.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_element_type.h"  // from @litert

namespace litert::lm {

SafetensorsDtypeInfo GetSafetensorsDtypeInfo(
    ::litert::ElementType element_type) {
  switch (element_type) {
    case ::litert::ElementType::Float32:
      return {"F32", sizeof(float)};
    case ::litert::ElementType::Float16:
      return {"F16", 2};
    case ::litert::ElementType::BFloat16:
      return {"BF16", 2};
    case ::litert::ElementType::Int32:
      return {"I32", sizeof(int32_t)};
    case ::litert::ElementType::Int64:
      return {"I64", sizeof(int64_t)};
    case ::litert::ElementType::Int8:
      return {"I8", sizeof(int8_t)};
    case ::litert::ElementType::UInt8:
      return {"U8", sizeof(uint8_t)};
    case ::litert::ElementType::Bool:
      return {"BOOL", sizeof(bool)};
    default:
      return {kDefaultSafetensorsDtype, sizeof(float)};
  }
}

std::string BuildSafetensorsHeader(
    absl::string_view tensor_name, absl::string_view dtype,
    const std::vector<int64_t>& shape, size_t total_bytes,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  nlohmann::ordered_json header;
  header[std::string(tensor_name)] = {
      {"dtype", std::string(dtype)},
      {"shape", shape},
      {"data_offsets", {0, total_bytes}},
  };
  if (!metadata.empty()) {
    header["__metadata__"] = metadata;
  }

  std::string json_header = header.dump();
  size_t remainder = json_header.size() % kSafetensorsHeaderAlignment;
  if (remainder != 0) {
    json_header.append(kSafetensorsHeaderAlignment - remainder, ' ');
  }
  return json_header;
}

}  // namespace litert::lm
