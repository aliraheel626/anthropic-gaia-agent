"""Command-line interface for GAIA Agent."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging with rich output."""
    handlers = [RichHandler(console=console, rich_tracebacks=True)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def cmd_run(args):
    """Run a single task."""
    from .config import get_config
    from .agent import run_single_task

    config = get_config()
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for e in errors:
            console.print(f"  - {e}")
        return 1

    console.print(f"[bold]Running task:[/bold] {args.question or args.task_id}")

    async def run():
        answer, state = await run_single_task(
            question=args.question,
            file_path=args.file,
            config=config,
        )

        if answer:
            console.print(f"\n[bold green]Answer:[/bold green] {answer}")
        else:
            console.print("\n[bold red]No answer generated[/bold red]")

        if state and state.last_error:
            console.print(f"[red]Error:[/red] {state.last_error}")

        return 0 if answer else 1

    return asyncio.run(run())


def cmd_benchmark(args):
    """Run benchmark evaluation."""
    from .config import get_config
    from .benchmark import run_benchmark as run_bench

    config = get_config()
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for e in errors:
            console.print(f"  - {e}")
        return 1

    async def run():
        result = await run_bench(
            split=args.split,
            level=args.level,
            max_tasks=args.max_tasks,
            data_dir=args.data_dir,
        )

        return 0 if result.accuracy >= 0.5 else 1

    return asyncio.run(run())


def cmd_stats(args):
    """Show dataset statistics."""
    from .benchmark import GAIALoader

    loader = GAIALoader(Path(args.data_dir))

    for split in ["validation", "test"]:
        stats = loader.get_stats(split)
        if stats["total"] > 0:
            console.print(f"\n[bold]{split.capitalize()} Set:[/bold]")
            console.print(f"  Total: {stats['total']}")
            console.print(f"  Level 1: {stats['level_1']}")
            console.print(f"  Level 2: {stats['level_2']}")
            console.print(f"  Level 3: {stats['level_3']}")
            console.print(f"  With files: {stats['with_files']}")

    return 0


def cmd_rag_add(args):
    """Add documents to RAG knowledge base."""
    from .tools.rag import get_rag_system

    rag = get_rag_system()

    path = Path(args.path)
    if path.is_file():
        content = path.read_text()
        success = rag.add_document(path.name, content, {"path": str(path)})
        if success:
            console.print(f"[green]Added document: {path.name}[/green]")
        else:
            console.print(f"[red]Failed to add document: {path.name}[/red]")
    elif path.is_dir():
        count = 0
        for file in path.glob(args.pattern or "*"):
            if file.is_file():
                try:
                    content = file.read_text()
                    if rag.add_document(file.name, content, {"path": str(file)}):
                        count += 1
                except Exception as e:
                    console.print(f"[yellow]Skipped {file.name}: {e}[/yellow]")
        console.print(f"[green]Added {count} documents[/green]")
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        return 1

    return 0


def cmd_rag_query(args):
    """Query RAG knowledge base."""
    from .tools.rag import get_rag_system

    rag = get_rag_system()
    results = rag.query(args.query, top_k=args.top_k)

    if not results:
        console.print("No results found.")
        return 0

    console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

    for i, doc in enumerate(results, 1):
        console.print(f"[cyan]{i}. {doc.id}[/cyan] (score: {doc.score:.3f})")
        console.print(f"   {doc.content[:200]}...")
        console.print()

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="gaia-agent",
        description="GAIA Benchmark Autonomous Agent",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single task")
    run_parser.add_argument("question", nargs="?", help="Task question")
    run_parser.add_argument("--task-id", help="Task ID from dataset")
    run_parser.add_argument("--file", "-f", help="Path to attached file")
    run_parser.set_defaults(func=cmd_run)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--split", default="validation", choices=["validation", "test"])
    bench_parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Difficulty level")
    bench_parser.add_argument("--max-tasks", type=int, default=20, help="Max tasks to run")
    bench_parser.add_argument("--data-dir", default="./data/gaia", help="GAIA dataset directory")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset stats")
    stats_parser.add_argument("--data-dir", default="./data/gaia", help="GAIA dataset directory")
    stats_parser.set_defaults(func=cmd_stats)

    # RAG commands
    rag_parser = subparsers.add_parser("rag", help="RAG knowledge base commands")
    rag_sub = rag_parser.add_subparsers(dest="rag_command")

    rag_add = rag_sub.add_parser("add", help="Add documents")
    rag_add.add_argument("path", help="File or directory path")
    rag_add.add_argument("--pattern", help="Glob pattern for directory")
    rag_add.set_defaults(func=cmd_rag_add)

    rag_query = rag_sub.add_parser("query", help="Query knowledge base")
    rag_query.add_argument("query", help="Search query")
    rag_query.add_argument("--top-k", type=int, default=5, help="Number of results")
    rag_query.set_defaults(func=cmd_rag_query)

    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
