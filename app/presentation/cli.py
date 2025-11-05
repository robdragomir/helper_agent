"""
CLI Interface - Presentation layer.
Handles user input and output formatting.
"""

import os
import typer
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Load settings first to get API keys
from app.core import settings

# Set Tavily API key in environment if it's configured but not already in env
if settings.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

from app.application import get_workflow
from app.infrastructure import KnowledgeBaseManager

app = typer.Typer(
    name="LangGraph Helper",
    help="A conversational AI helper for LangGraph documentation",
)

console = Console()


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question about LangGraph"),
    mode: str = typer.Option(
        ...,
        "--mode",
        "-m",
        help="Search mode: offline (local KB), online (web search), or both (combine sources)",
    ),
) -> None:
    """
    Ask a question about LangGraph.

    You must specify a search mode:
    - offline: Search only in the local knowledge base
    - online: Search the web for latest information
    - both: Search both sources and combine results

    Examples:
        helper-agent ask "How do I create a basic graph?" --mode offline
        helper-agent ask "What are the latest features?" --mode online
        helper-agent ask "How does state work?" --mode both
    """
    # Validate mode
    if mode not in ["offline", "online", "both"]:
        console.print(
            "[red]Error: Mode must be 'offline', 'online', or 'both'[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Processing: {query}[/cyan]")
    console.print()

    try:
        # Get workflow and run
        workflow = get_workflow()
        answer = workflow.run(query, mode=mode)

        # Display answer
        console.print(Panel(answer.text, title="Answer", border_style="green"))
        console.print()

        # Display metadata
        metadata_table = Table(title="Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="green")

        metadata_table.add_row(
            "Confidence", f"{answer.answer_confidence:.2%}"
        )
        metadata_table.add_row("Offline Sources", "✓" if answer.used_offline else "✗")
        metadata_table.add_row("Online Sources", "✓" if answer.used_online else "✗")

        console.print(metadata_table)
        console.print()

        # Display citations
        if answer.citations:
            console.print("[bold]Sources:[/bold]")
            for citation in answer.citations:
                console.print(f"  {citation['label']} {citation['source']} ({citation['note']})")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def build_kb(
    force_rebuild: bool = typer.Option(
        False,
        "--force-rebuild",
        "-f",
        help="Force rebuild everything without checking for changes"
    )
) -> None:
    """
    Build or rebuild the knowledge base from multiple sources.

    This downloads the latest LangGraph and LangChain documentation,
    detects changes, and creates embeddings.

    By default, it checks if documents have changed and only rebuilds if needed.
    Use --force-rebuild to skip change detection and rebuild everything.
    """
    console.print("[cyan]Building knowledge base from multiple sources (LangGraph + LangChain)...[/cyan]")

    try:
        kb_manager = KnowledgeBaseManager()
        success = kb_manager.build_kb(force_rebuild=force_rebuild)

        if success:
            console.print("[green]✓ Knowledge base built successfully[/green]")
        else:
            console.print("[red]✗ Failed to build knowledge base[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """Show telemetry and performance statistics."""
    from app.infrastructure.telemetry import TelemetryLogger, EvaluationMetrics

    console.print("[cyan]Loading telemetry data...[/cyan]")

    try:
        telemetry = TelemetryLogger()
        eval_metrics = EvaluationMetrics()

        # Get recent traces
        traces = telemetry.get_recent_traces(limit=100)
        avg_metrics = eval_metrics.get_average_metrics(limit=100)

        # Display summary
        if traces:
            console.print(f"\n[bold]Recent Traces: {len(traces)}[/bold]")

        if avg_metrics:
            metrics_table = Table(title="Average Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            for metric_name, metric_value in avg_metrics.items():
                if isinstance(metric_value, float):
                    metrics_table.add_row(metric_name, f"{metric_value:.3f}")
                else:
                    metrics_table.add_row(metric_name, str(metric_value))

            console.print(metrics_table)

        if not traces and not avg_metrics:
            console.print("[yellow]No telemetry data available yet[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive(
    mode: str = typer.Option(
        "both",
        "--mode",
        "-m",
        help="Search mode: offline, online, or both",
    ),
) -> None:
    """
    Start interactive chat mode.

    Uses the specified search mode for all queries in the conversation.
    Default mode is 'both' (search both local KB and web).
    """
    # Validate mode
    if mode not in ["offline", "online", "both"]:
        console.print(
            "[red]Error: Mode must be 'offline', 'online', or 'both'[/red]"
        )
        raise typer.Exit(1)

    console.print(
        "[cyan bold]LangGraph Helper - Interactive Mode[/cyan bold]"
    )
    console.print(f"[dim]Mode: {mode} | Type 'exit' to quit[/dim]\n")

    workflow = get_workflow()

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ")

            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not query.strip():
                continue

            console.print()
            answer = workflow.run(query, mode=mode)

            console.print(f"[bold green]Assistant:[/bold green]")
            console.print(answer.text)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]\n")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()