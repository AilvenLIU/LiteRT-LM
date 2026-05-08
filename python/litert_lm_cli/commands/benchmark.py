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

"""Benchmark subcommand for LiteRT-LM CLI."""

import click

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import model


@click.command(
    cls=help_formatter.ColorCommand,
    help="""Benchmarks a LiteRT-LM model.
  \b
  Examples:
    # Benchmark using a model ID from 'litert-lm list'
    litert-lm benchmark my-model

    # Benchmark using a local path
    litert-lm benchmark ./model.litertlm

    # Benchmark directly from a HuggingFace repository
    litert-lm benchmark --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference")
@click.option(
    "-p",
    "--prefill-tokens",
    default=256,
    type=int,
    help="The number of tokens to prefill.",
)
@click.option(
    "-d",
    "--decode-tokens",
    default=256,
    type=int,
    help="The number of tokens to decode.",
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=None,
    help=(
        "Maximum number of tokens for the KV cache. If not set, it will be"
        " chosen based on --prefill_tokens and --decode_tokens."
    ),
)
@common.common_inference_options
def benchmark(
    model_reference: str,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    backend: str = "cpu",
    android: bool = False,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
    max_num_tokens: int | None = None,
    npu_library_dir: str = "",
):
  """Benchmarks a LiteRT-LM model.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prefill_tokens: The number of tokens to prefill.
    decode_tokens: The number of tokens to decode.
    backend: The backend to use (cpu, gpu or npu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    npu_library_dir: The directory containing NPU libraries.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  if from_huggingface_repo:
    model_path = common.download_from_huggingface(
        from_huggingface_repo, model_reference, huggingface_token
    )
    if not model_path:
      return
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)

  if max_num_tokens is None:
    # Replicates the logic from
    # runtime/engine/engine_settings.cc
    max_num_tokens = ((prefill_tokens + 1023) // 4096 + 1) * 4096

  model_obj.benchmark(
      prefill_tokens=prefill_tokens,
      decode_tokens=decode_tokens,
      is_android=android,
      backend=backend,
      enable_speculative_decoding=enable_speculative_decoding,
      max_num_tokens=max_num_tokens,
      npu_library_dir=npu_library_dir,
  )


def register(cli: click.Group) -> None:
  """Registers the benchmark command."""
  cli.add_command(benchmark)
