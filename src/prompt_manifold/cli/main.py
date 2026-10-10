"""Production CLI for PromptManifold powered by Typer and Rich."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompt_manifold import __version__
from prompt_manifold.core.engine import PromptManifoldEngine
from prompt_manifold.eval.benchmark import BenchmarkRunner
from prompt_manifold.schemas import ExpansionConfig, ManifoldTarget

app = typer.Typer(
    name="manifold",
    help="PromptManifold: Inference-Time Prompt Distribution Alignment Engine",
    add_completion=False,
)
console = Console()
engine = PromptManifoldEngine()


@app.command()
def enhance(
    prompt: str = typer.Argument(..., help="Sparse user prompt to project onto manifold"),
    target: str = typer.Option(
        "flux", "--target", "-t", help="Target model (flux, sdxl, sd3, midjourney, dalle3)"
    ),
    detail: int = typer.Option(3, "--detail", "-d", help="Expansion detail level (1-5)"),
    negative: bool = typer.Option(
        True, "--negative/--no-negative", help="Synthesize paired negative prompt"
    ),
):
    """Project a sparse prompt onto the model's in-distribution training manifold."""
    try:
        target_model = ManifoldTarget(target.lower())
    except ValueError:
        console.print(
            f"[bold red]Error:[/] Unknown target model '{target}'. "
            "Choose from flux, sdxl, sd3, midjourney, dalle3."
        )
        raise typer.Exit(code=1)

    cfg = ExpansionConfig(
        target_model=target_model,
        detail_level=detail,
        generate_negative=negative,
    )
    res = engine.enhance(prompt, cfg)

    console.print(Panel(
        f"[bold white]{res.enhanced_prompt}[/]",
        title=f"[cyan]PromptManifold Projection ({target_model.value.upper()})[/]",
        border_style="cyan",
    ))

    if res.negative_prompt:
        console.print(Panel(
            f"[dim red]{res.negative_prompt}[/]",
            title="[red]Dual Manifold Negative Constraint[/]",
            border_style="red",
        ))

    table = Table(title="Inference Telemetry", border_style="dim white", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_column("Status", style="dim")

    raw_words = len(prompt.split())
    enh_words = len(res.enhanced_prompt.split())

    table.add_row(
        "In-Distribution Proximity",
        f"{res.in_distribution_score:.3f}",
        "Optimal [0.7 - 1.0]"
    )
    table.add_row(
        "Expansion Ratio",
        f"{res.expansion_ratio:.2f}x",
        f"{raw_words} -> {enh_words} tokens"
    )
    table.add_row(
        "Semantic Drift Penalty",
        f"{res.semantic_drift_score:.3f}",
        "Preserved [<= 0.15]"
    )
    table.add_row(
        "Slot Coverage Ratio",
        f"{res.slot_coverage_score:.3f}",
        "High Density [>= 0.8]"
    )
    table.add_row(
        "Projection Latency",
        f"{res.latency_ms:.2f} ms",
        "Sub-millisecond Engine"
    )

    console.print(table)


@app.command()
def inspect(
    prompt: str = typer.Argument(..., help="Prompt to decompose into semantic slots"),
):
    """Inspect syntactic slot decomposition without generating final prompt."""
    slots = engine.extractor.extract(prompt)

    table = Table(title=f"Semantic Slot Decomposition: '{prompt}'", border_style="dim white")
    table.add_column("Slot", style="bold cyan")
    table.add_column("Extracted Content", style="white")

    table.add_row("Primary Subject", slots.subject)
    attrs = ", ".join(slots.subject_attributes) if slots.subject_attributes else "[dim]None[/]"
    table.add_row("Attributes", attrs)
    table.add_row("Medium", slots.medium.value)
    table.add_row("Lighting", slots.lighting or "[dim]Unspecified (Inferred on Projection)[/]")
    table.add_row(
        "Camera / Optics",
        slots.camera_framing or "[dim]Unspecified (Inferred on Projection)[/]"
    )
    pres = ", ".join(slots.preserved_tokens) if slots.preserved_tokens else "[dim]None[/]"
    table.add_row("Preserved Tokens", pres)
    neg = ", ".join(slots.explicit_negatives) if slots.explicit_negatives else "[dim]None[/]"
    table.add_row("Explicit Negatives", neg)

    console.print(table)


@app.command()
def benchmark():
    """Run empirical benchmark comparing Naive baseline vs PromptManifold."""
    runner = BenchmarkRunner(engine)
    with console.status("[bold cyan]Executing PromptManifold empirical benchmark suite..."):
        report = runner.run_suite()

    table = Table(
        title="PromptManifold Benchmark Report (Naive Standard vs Enhanced Manifold)",
        border_style="cyan"
    )
    table.add_column("ID", style="dim")
    table.add_column("Raw Prompt", style="white")
    table.add_column("Target", style="cyan")
    table.add_column("Naive S_ID", style="red")
    table.add_column("Enhanced S_ID", style="green")
    table.add_column("Expansion", style="yellow")
    table.add_column("Drift", style="magenta")
    table.add_column("Latency (ms)", style="blue")

    for sc in report["scenarios"]:
        table.add_row(
            sc["id"],
            sc["raw_prompt"],
            sc["target"].upper(),
            f"{sc['naive_id_score']:.2f}",
            f"{sc['enhanced_id_score']:.2f}",
            f"{sc['expansion_ratio']:.1f}x",
            f"{sc['semantic_drift']:.2f}",
            f"{sc['latency_ms']:.2f}",
        )

    console.print(table)

    summary_panel = Panel(
        f"[bold]Scenarios Evaluated:[/] {report['scenario_count']}\n"
        f"[bold]Mean Naive In-Dist Score:[/] [red]{report['mean_naive_id_score']:.3f}[/]\n"
        f"[bold]Mean Enhanced In-Dist Score:[/] [green]{report['mean_enhanced_id_score']:.3f}[/]\n"
        f"[bold]Mean Expansion Ratio:[/] [yellow]{report['mean_expansion_ratio']:.2f}x[/]\n"
        f"[bold]Mean Projection Latency:[/] [blue]{report['mean_latency_ms']:.2f} ms[/]\n"
        f"[bold]Manifold Win Rate:[/] [bold green]{report['win_rate_percentage']:.1f}%[/]",
        title="[bold green]Empirical Summary[/]",
        border_style="green",
    )
    console.print(summary_panel)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable live auto-reload"),
):
    """Start the PromptManifold production FastAPI service and Developer Console."""
    import uvicorn
    console.print(f"[bold green]Starting PromptManifold API service at http://{host}:{port}[/]")
    uvicorn.run("prompt_manifold.api.app:app", host=host, port=port, reload=reload)


@app.command()
def version():
    """Display engine version information."""
    console.print(f"PromptManifold v{__version__} [Inference-Time Prompt Distribution Engine]")


if __name__ == "__main__":
    app()
