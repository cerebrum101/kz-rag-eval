"""ai-eval CLI. Single command: `ai-eval run <config.yaml>`."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ai_eval_harness import __version__
from ai_eval_harness.config import EvalConfig
from ai_eval_harness.runner import run_eval


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="ai-eval")
def main() -> None:
    """ai-eval-harness — open-source RAG and agent eval harness."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs"),
    help="Directory to write the run report. Default: ./runs",
)
@click.option(
    "--no-score",
    is_flag=True,
    help="Skip Ragas scoring. Useful when judge-model creds aren't available.",
)
def run(config_path: Path, out: Path, no_score: bool) -> None:
    """Run an eval defined by CONFIG_PATH (YAML)."""
    console = Console()
    console.rule(f"[bold]ai-eval-harness v{__version__}[/bold]")
    console.print(f"[dim]config:[/dim] {config_path}")

    cfg = EvalConfig.from_yaml(config_path)
    console.print(f"[dim]models:[/dim] {', '.join(m.name for m in cfg.models)}")

    result = run_eval(cfg, score=not no_score)
    report_path = result.write(out)

    _print_cost_table(console, result.cost)
    _print_score_table(console, result.deterministic_scores, title="Deterministic scores")
    if result.ragas_scores:
        _print_score_table(console, result.ragas_scores, title="Ragas scores")
    else:
        console.print(
            "[yellow]Ragas scores: not computed[/yellow] "
            "(judge-model credentials missing, --no-score, or the ragas extra not installed)"
        )

    if result.promptfoo_scores:
        _print_score_table(console, result.promptfoo_scores, title="promptfoo pass-rates")
    elif result.promptfoo_note:
        console.print(f"[yellow]promptfoo:[/yellow] {result.promptfoo_note}")

    console.print(f"[green]Report:[/green] {report_path}")


def _print_cost_table(console: Console, summaries: list) -> None:  # type: ignore[type-arg]
    table = Table(title="Cost + latency", show_lines=False)
    table.add_column("model", style="cyan")
    table.add_column("requests", justify="right")
    table.add_column("failures", justify="right")
    table.add_column("input tok", justify="right")
    table.add_column("output tok", justify="right")
    table.add_column("avg latency ms", justify="right")
    table.add_column("cost USD", justify="right", style="green")

    for s in summaries:
        table.add_row(
            s.model_name,
            str(s.requests),
            str(s.failures),
            f"{s.input_tokens:,}",
            f"{s.output_tokens:,}",
            f"{s.avg_latency_ms:.1f}",
            f"${s.cost_usd:.4f}",
        )
    console.print(table)


def _print_score_table(
    console: Console,
    scores: dict[str, dict[str, float]],
    title: str = "Scores",
) -> None:
    metrics_seen: list[str] = []
    for per_model in scores.values():
        for k in per_model:
            if k not in metrics_seen and not k.startswith("_"):
                metrics_seen.append(k)

    table = Table(title=title, show_lines=False)
    table.add_column("model", style="cyan")
    for m in metrics_seen:
        table.add_column(m, justify="right")

    for model_name, per_model in scores.items():
        row = [model_name]
        for m in metrics_seen:
            v = per_model.get(m)
            row.append(f"{v:.3f}" if isinstance(v, float) else "-")
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
