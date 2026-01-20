from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich import print

from lepto_ews.config import AppConfig
from lepto_ews.pipeline import run_end_to_end

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to config.yaml (if provided without a subcommand, runs the pipeline)",
    ),
):
    """Leptospirosis EWS CLI."""
    if ctx.invoked_subcommand is None and config is not None:
        run(config=config)


@app.command("run")
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="Path to config.yaml"),
):
    """Run the full EWS pipeline (features → model → SHAP → GIS maps)."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = AppConfig.model_validate(raw)
    print(f"Running pipeline. Output dir: {cfg.output_dir}")
    run_end_to_end(cfg)
    print("Done.")


if __name__ == "__main__":
    app()
