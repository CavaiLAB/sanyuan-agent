"""NeuroAgent Harness CLI."""

from pathlib import Path

import typer

from . import __version__
from .local_desktop.probe import run_probe
from .local_desktop.stdio_guard import run_guarded_desktop_process

app = typer.Typer(help="NeuroAgent Harness CLI")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"neuroagent {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version", callback=_version_callback, is_eager=True
    ),
) -> None:
    """NeuroAgent Harness — Agent engine CLI."""


@app.command()
def run(config: Path) -> None:
    """Run agent from a config file (stub)."""
    typer.echo(f"Config path: {config}")


@app.command("desktop-run")
def desktop_run() -> None:
    """Run one validated desktop payload from stdin and emit SSE on stdout."""

    exit_code = run_guarded_desktop_process()
    if exit_code:
        raise typer.Exit(exit_code)


@app.command("sandbox-probe", hidden=True)
def sandbox_probe(
    mode: str,
    nonce: str,
    inside: Path,
    outside: Path | None = None,
) -> None:
    """Internal behavior probe used by the packaged desktop sidecar."""

    raise typer.Exit(run_probe(mode, nonce, inside, outside))


if __name__ == "__main__":
    app()
