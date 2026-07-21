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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "levenshtein_align",
    srcs = ["levenshtein_align.cc"],
    hdrs = ["levenshtein_align.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "levenshtein_align_test",
    srcs = ["levenshtein_align_test.cc"],
    deps = [
        ":levenshtein_align",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "token_merger",
    srcs = ["token_merger.cc"],
    hdrs = ["token_merger.h"],
    deps = [
        ":levenshtein_align",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "token_merger_test",
    srcs = ["token_merger_test.cc"],
    deps = [
        ":token_merger",
        "@com_google_googletest//:gtest_main",
    ],
)
