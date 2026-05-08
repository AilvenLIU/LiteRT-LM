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

"""Utility functions for litert-lm models."""

import dataclasses
import glob
import importlib.util
import inspect
import mimetypes
import os
import pathlib
import traceback

import click

import litert_lm

try:
  # pylint: disable=g-import-not-at-top
  from litert_lm.adb import adb_benchmark  # pytype: disable=import-error

  _HAS_ADB = True
except ImportError:
  _HAS_ADB = False


def get_attachment_type(path: str) -> str:
  """Returns the attachment type (audio or image) from the file path.

  Args:
    path: Path to the attachment.

  Returns:
    'audio' or 'image'.

  Raises:
    ValueError: If the file type cannot be determined or is unsupported.
  """
  mime_type, _ = mimetypes.guess_type(path)
  if mime_type:
    if mime_type.startswith("audio/"):
      return "audio"
    elif mime_type.startswith("image/"):
      return "image"
    else:
      raise ValueError(f"Unsupported attachment type for '{path}': {mime_type}")
  else:
    raise ValueError(f"Could not determine file type for attachment '{path}'.")


def load_preset(preset: str):
  """Loads a preset file and returns the tools, messages and extra_context."""
  click.echo(click.style(f"Loading preset from {preset}:", dim=True))
  if not os.path.exists(preset):
    click.echo(click.style(f"Preset file not found: {preset}", fg="red"))
    return None, None, None

  spec = importlib.util.spec_from_file_location("user_tools", preset)
  if not spec or not spec.loader:
    click.echo(click.style(f"Failed to load tools from {preset}", fg="red"))
    return None, None, None

  user_tools = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(user_tools)

  tools = getattr(user_tools, "tools", None)
  if tools is None:
    tools = [
        obj
        for name, obj in inspect.getmembers(user_tools, inspect.isfunction)
        if obj.__module__ == "user_tools"
    ]

  messages = None
  system_instruction = getattr(user_tools, "system_instruction", None)
  if system_instruction:
    click.echo(
        click.style(f"- System instruction: {system_instruction}", dim=True)
    )
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}],
    }]

  click.echo(click.style("- Tools:", dim=True))
  for tool in tools:
    click.echo(
        click.style(f"  - {getattr(tool, '__name__', str(tool))}", dim=True)
    )

  extra_context = getattr(user_tools, "extra_context", None)
  if extra_context:
    click.echo(click.style(f"- Extra context: {extra_context}", dim=True))

  return tools, messages, extra_context


def _parse_backend(
    backend: str, npu_library_dir: str = ""
) -> litert_lm.Backend:
  """Parses the backend string and returns the corresponding Backend enum."""
  backend_lower = backend.lower()
  if backend_lower == "gpu":
    return litert_lm.Backend.GPU()
  if backend_lower == "npu":
    return litert_lm.Backend.NPU(native_library_dir=npu_library_dir)
  return litert_lm.Backend.CPU()


@dataclasses.dataclass
class Model:
  """Represents a LiteRT-LM model.

  Attributes:
    model_id: The ID of the model.
    model_path: The local path to the model file.
  """

  model_id: str
  model_path: str

  def exists(self) -> bool:
    """Returns True if the model file exists locally."""
    return os.path.isfile(self.model_path)

  def to_str(self) -> str:
    """Returns a string representation of the model."""
    return self.model_id

  def benchmark(
      self,
      prefill_tokens: int = 256,
      decode_tokens: int = 256,
      is_android: bool = False,
      backend: str = "cpu",
      enable_speculative_decoding: bool | None = None,
      max_num_tokens: int | None = None,
      npu_library_dir: str = "",
  ):
    """Benchmarks the model.

    Args:
      prefill_tokens: The number of tokens to prefill.
      decode_tokens: The number of tokens to decode.
      is_android: Whether to run the benchmark on an Android device via ADB.
      backend: The backend to use (cpu, gpu or npu).
      enable_speculative_decoding: Whether to enable speculative decoding. If
        None, use the model's default.
      max_num_tokens: Maximum number of tokens for the KV cache.
      npu_library_dir: The directory containing NPU libraries.
    """
    if not self.exists():
      click.echo(
          click.style(
              f"Could not find {self.to_str()} locally in {self.model_path}.",
              fg="red",
          )
      )
      return

    try:
      backend_val = _parse_backend(backend, npu_library_dir)
      cache_dir_val = (
          ":memory"
          if isinstance(backend_val, litert_lm.Backend.CPU)
          else ":nocache"
      )

      if is_android:
        if not _HAS_ADB:
          raise ImportError("litert_lm.adb dependencies are not available.")
        benchmark_obj = adb_benchmark.AdbBenchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            max_num_tokens=max_num_tokens,
        )
      else:
        benchmark_obj = litert_lm.Benchmark(
            self.model_path,
            backend=backend_val,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            cache_dir=cache_dir_val,
            enable_speculative_decoding=enable_speculative_decoding,
            max_num_tokens=max_num_tokens,
        )

      click.echo(f"Benchmarking model: {self.to_str()} ({self.model_path})")
      click.echo(f"Backend                    : {backend}")
      click.echo(f"Number of tokens in prefill: {prefill_tokens}")
      click.echo(f"Number of tokens in decode : {decode_tokens}")
      if max_num_tokens is not None:
        click.echo(f"Max number of tokens       : {max_num_tokens}")

      spec_dec_str = "auto"
      if enable_speculative_decoding is True:
        spec_dec_str = "true"
      elif enable_speculative_decoding is False:
        spec_dec_str = "false"
      click.echo(f"Speculative decoding       : {spec_dec_str}")
      if is_android:
        click.echo("Target                     : Android")

      result = benchmark_obj.run()

      click.echo("----- Results -----")
      click.echo(
          f"Prefill speed:        {result.last_prefill_tokens_per_second:.2f}"
          " tokens/s"
      )
      click.echo(
          f"Decode speed:         {result.last_decode_tokens_per_second:.2f}"
          " tokens/s"
      )
      click.echo(f"Init time:            {result.init_time_in_second:.4f} s")
      click.echo(
          f"Time to first token:  {result.time_to_first_token_in_second:.4f} s"
      )

    except Exception:  # pylint: disable=broad-exception-caught
      click.echo(click.style("An error occurred during benchmarking", fg="red"))
      traceback.print_exc()

  @classmethod
  def get_all_models(cls):
    """Returns a list of all locally available models."""
    model_paths = glob.glob(
        "*/model.litertlm",
        root_dir=get_converted_models_base_dir(),
        recursive=True,
    )

    return [
        Model.from_model_id(pathlib.Path(path).parent.name.replace("--", "/"))
        for path in model_paths
    ]

  @classmethod
  def from_model_reference(cls, model_reference):
    """Creates a Model instance from a model reference."""
    if os.path.exists(model_reference):
      return cls.from_model_path(model_reference)
    else:
      # assume the reference is model_id
      return cls.from_model_id(model_reference)

  @classmethod
  def from_model_path(cls, model_path):
    """Creates a Model instance from a model path."""
    return cls(
        model_id=os.path.basename(model_path),
        model_path=os.path.abspath(model_path),
    )

  @classmethod
  def from_model_id(cls, model_id):
    """Creates a Model instance from a model ID."""
    return cls(
        model_id=model_id,
        model_path=os.path.join(
            get_converted_models_base_dir(),
            model_id.replace("/", "--"),
            "model.litertlm",
        ),
    )


# Just to use the huggingface convention. Likely to change.
def model_id_dir_name(model_id):
  """Converts a model ID to a directory name."""
  return model_id.replace("/", "--")


# ~/.litert-lm/models
def get_converted_models_base_dir():
  """Gets the base directory for all converted models."""
  return os.path.join(os.path.expanduser("~"), ".litert-lm", "models")


# ~/.litert-lm/models/<model_id>
def get_model_dir(model_id):
  """Gets the model directory for a given model ID."""
  return os.path.join(
      get_converted_models_base_dir(),
      model_id_dir_name(model_id),
  )
