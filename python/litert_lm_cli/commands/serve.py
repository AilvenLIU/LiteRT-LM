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

"""HTTP server for LiteRT-LM with Gemini-compatible API.

Reference: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

import http.server

import click

import litert_lm
from litert_lm_cli import help_formatter
from litert_lm_cli.commands import gemini_handler
from litert_lm_cli.commands import openai_handler
from litert_lm_cli.commands import serve_util


def run_server(
    host: str,
    port: int,
    handler_class: type[http.server.BaseHTTPRequestHandler],
    api_name: str,
) -> None:
  """Starts the HTTP server.

  Args:
    host: Host to listen on.
    port: Port to listen on.
    handler_class: The HTTP handler class to use.
    api_name: The API protocol name (e.g., "OpenAI", "Gemini").
  """
  server_address = (host, port)
  article = "an" if api_name.lower()[0] in "aeiou" else "a"
  try:
    with serve_util.LiteRTLMServer(server_address, handler_class) as server:
      click.echo(
          click.style(
              f"Starting {article} {api_name}-compatible API server on"
              f" {host}:{port}...",
              fg="green",
              bold=True,
          )
      )
      try:
        server.serve_forever()
      finally:
        if server.litert_lm_engine is not None:
          server.litert_lm_engine.__exit__(None, None, None)
  except KeyboardInterrupt:
    click.echo(click.style("\nShutting down server...", fg="cyan"))


@click.command(
    cls=help_formatter.ColorCommand,
    help=(
        "Start a server with an OpenAI or Gemini-compatible API"
        " (alpha feature)\n\n"
        "This server hosts locally imported LiteRT-LM models.\n"
        '  - Use "litert-lm import" to import a new model.\n'
        '  - Use "litert-lm list" to view already imported models.\n\n'
        "Supported OpenAI endpoints:\n"
        "  - /v1/models\n"
        "  - /v1/chat/completions\n\n"
        "Supported Gemini endpoints:\n"
        "  - /v1beta/models/{model_spec}:generateContent\n"
        "  - /v1beta/models/{model_spec}:streamGenerateContent"
    ),
)
@click.option("--host", default="0.0.0.0", type=str, help="Host to listen on")
@click.option("--port", default=9379, type=int, help="Port to listen on")
@click.option(
    "--api",
    type=click.Choice(["openai", "gemini"], case_sensitive=False),
    default="openai",
    help="The API protocol to use.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def serve(host: str, port: int, *, api: str, verbose: bool) -> None:
  """Starts a local HTTP server speaking the OpenAI or Gemini API protocol.

  This server hosts locally imported LiteRT-LM models.
    - Use "litert-lm import" to import a new model.
    - Use "litert-lm list" to view already imported models.

  Supported OpenAI endpoints:
    - /v1/models
    - /v1/chat/completions
    - /v1/responses

  Supported Gemini endpoints:
    - /v1beta/models/{model_spec}:generateContent
    - /v1beta/models/{model_spec}:streamGenerateContent

  Args:
    host: Host to listen on.
    port: Port to listen on.
    api: The API protocol to use (openai or gemini).
    verbose: Whether to enable verbose logging.
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  api_lower = api.lower()
  if api_lower == "gemini":
    handler_class = gemini_handler.GeminiHandler
    api_name = "Gemini"
  elif api_lower == "openai":
    handler_class = openai_handler.OpenAIHandler
    api_name = "OpenAI"
  else:
    raise click.BadParameter(f"Unsupported API: {api}")

  run_server(host, port, handler_class, api_name)


def register(cli: click.Group) -> None:
  """Registers the serve command."""
  cli.add_command(serve)
