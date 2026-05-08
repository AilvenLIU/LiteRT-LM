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

"""Run subcommand for LiteRT-LM CLI."""

import os
import sys

import click

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import model
from litert_lm_cli.commands import convert as _convert_module


@click.command(
    cls=help_formatter.ColorCommand,
    help="""Runs a LiteRT-LM model interactively or with a single prompt.
  \b
  Examples:
    # Run interactively using a model ID from 'litert-lm list'
    litert-lm run my-model

    # Run with a single prompt using a local path
    litert-lm run ./model.litertlm --prompt "Hi there!"

    # Run directly from a HuggingFace repository
    litert-lm run --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference")
@click.option(
    "--prompt", default=None, help="A single prompt to run once and exit."
)
@click.option(
    "--preset",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a Python file containing tool functions and system"
        " instructions."
    ),
)
@click.option(
    "--no-template",
    is_flag=True,
    default=False,
    help=(
        "Interact with the model directly without applying prompt templates."
        " That means the input should include all control tokens for the model"
        " expected."
    ),
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=None,
    help=(
        "Maximum number of tokens for the KV cache. If not set, use the"
        " default from the native engine."
    ),
)
@click.option(
    "--filter-channel-content-from-kv-cache",
    is_flag=True,
    default=False,
    help="Whether to filter channel content from the KV cache.",
)
@click.option(
    "--vision-backend",
    type=click.Choice(["cpu", "gpu", ""], case_sensitive=False),
    default=None,
    help="The backend to use for vision encoding.",
)
@click.option(
    "--audio-backend",
    type=click.Choice(["cpu", "gpu", ""], case_sensitive=False),
    default=None,
    help="The backend to use for audio encoding.",
)
@click.option(
    "--attachment",
    multiple=True,
    type=click.Path(dir_okay=False),
    help=(
        "Path to an attachment (image or audio only). Can be specified multiple"
        " times. Attachements are placed before the first user text prompt."
    ),
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "The number of top logits used during sampling. If not set, use the"
        " default from the model or engine."
    ),
)
@click.option(
    "--top-p",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "The cumulative probability threshold for nucleus sampling. If not set,"
        " use the default from the model or engine."
    ),
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=None,
    help=(
        "The temperature to use for sampling. If not set, use the default from"
        " the model or engine."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=(
        "The seed to use for randomization. If not set, use the default from"
        " the model or engine."
    ),
)
@common.common_inference_options
def run(
    model_reference,
    prompt=None,
    preset=None,
    backend="cpu",
    android=False,
    enable_speculative_decoding=None,
    verbose=False,
    no_template=False,
    from_huggingface_repo=None,
    huggingface_token=None,
    max_num_tokens=None,
    filter_channel_content_from_kv_cache=False,
    vision_backend=None,
    audio_backend=None,
    attachment=(),
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    npu_library_dir: str = "",
):
  r"""Runs a LiteRT-LM model interactively or with a single prompt.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prompt: A single prompt to run once and exit.
    preset: Path to a Python file containing tool functions and system
      instructions.
    backend: The backend to use (cpu or gpu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    no_template: Interact with the model directly without applying prompt
      templates or stripping stop tokens.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    filter_channel_content_from_kv_cache: Whether to filter channel content from
      the KV cache.
    vision_backend: The backend to use for vision tasks.
    audio_backend: The backend to use for audio tasks.
    attachment: Path to an attachment (e.g., image or audio).
    top_k: The number of top logits used during sampling.
    top_p: The cumulative probability threshold for nucleus sampling.
    temperature: The temperature to use for sampling.
    seed: The seed to use for randomization.
    npu_library_dir: The directory containing NPU libraries.
  """
  if attachment and no_template:
    click.echo(
        click.style(
            "Error: Attachments are not supported with --no-template.",
            fg="red",
        )
    )
    return

  expanded_attachments = []
  has_audio = False
  has_image = False

  for a in attachment:
    expanded = os.path.expanduser(a)
    if not os.path.exists(expanded):
      raise click.BadParameter(f"File '{a}' does not exist.")
    expanded_attachments.append(expanded)

    try:
      a_type = model.get_attachment_type(expanded)
      if a_type == "audio":
        has_audio = True
      elif a_type == "image":
        has_image = True
    except ValueError as e:
      raise click.BadParameter(str(e)) from e

  if has_audio and not audio_backend:
    click.echo(
        click.style(
            "Error: Audio attachments require --audio-backend to be set.",
            fg="red",
        )
    )
    return

  if has_image and not vision_backend:
    click.echo(
        click.style(
            "Error: Image attachments require --vision-backend to be set.",
            fg="red",
        )
    )
    return

  # If the stdin is not connected to the terminal, e.g., piped or redirected
  # input, then handle the input as the one-shot prompt.
  #
  # # Redirected input:
  # $ litert-lm run < prompt.txt
  # $ litert-lm run --prompt="Explain this error log" < error.log
  #
  # # Piped input:
  # $ cat text.txt | litert-lm run --prompt="Summarize the content."
  if not sys.stdin.isatty():
    piped_input = sys.stdin.read().strip()
    if piped_input:
      prompt = f"{prompt}\n\n{piped_input}" if prompt else piped_input
    elif not prompt:
      # If no prompt is provided and it's not a TTY, we can't be interactive.
      return

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
    if not model_obj.exists():
      # Only auto-convert if it looks like a HuggingFace repo ID (account/repo)
      # and is not a local path.
      parts = model_reference.split("/")
      if len(parts) == 2 and all(parts) and not os.path.exists(model_reference):
        click.echo(
            click.style(
                f"Model '{model_reference}' not found. Attempting to convert"
                f" from https://huggingface.co/{model_reference} ...",
                fg="yellow",
            )
        )
        model_obj = model.Model.from_model_reference(model_reference)

      if not model_obj.exists():
        click.echo(
            click.style(
                f"Failed to find or convert model '{model_reference}'.",
                fg="red",
            )
        )
        return

  model_obj.run_interactive(
      prompt=prompt,
      is_android=android,
      backend=backend,
      preset=preset,
      enable_speculative_decoding=enable_speculative_decoding,
      no_template=no_template,
      max_num_tokens=max_num_tokens,
      filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
      vision_backend=vision_backend,
      audio_backend=audio_backend,
      attachments=tuple(expanded_attachments),
      top_k=top_k,
      top_p=top_p,
      temperature=temperature,
      seed=seed,
      npu_library_dir=npu_library_dir,
  )


def register(cli: click.Group) -> None:
  """Registers the run command."""
  cli.add_command(run)
