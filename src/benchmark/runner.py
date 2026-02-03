"""Benchmark Runner.

Executes GAIA benchmark tasks and collects results.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .gaia_loader import GAIALoader, GAIATask
from .evaluator import Evaluator, EvaluationResult
from ..agent import GAIAAgent
from ..config import Config, get_config
from ..planning import TaskState

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TaskResult:
    """Result of running a single task."""

    task: GAIATask
    agent_answer: Optional[str]
    evaluation: Optional[EvaluationResult]
    state: Optional[TaskState]
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark run results."""

    split: str
    level: Optional[int]
    total_tasks: int
    completed_tasks: int
    correct_tasks: int
    failed_tasks: int
    accuracy: float
    task_results: list[TaskResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_result(self, result: TaskResult) -> None:
        """Add a task result."""
        self.task_results.append(result)
        self.completed_tasks += 1

        if result.error:
            self.failed_tasks += 1
        elif result.evaluation and result.evaluation.is_correct:
            self.correct_tasks += 1

        # Update accuracy
        if self.completed_tasks > 0:
            self.accuracy = self.correct_tasks / self.completed_tasks

    def get_level_stats(self) -> dict[int, dict]:
        """Get statistics by difficulty level."""
        stats = {
            1: {"total": 0, "correct": 0},
            2: {"total": 0, "correct": 0},
            3: {"total": 0, "correct": 0},
        }

        for result in self.task_results:
            level = result.task.level
            stats[level]["total"] += 1
            if result.evaluation and result.evaluation.is_correct:
                stats[level]["correct"] += 1

        # Add accuracy
        for level in stats:
            total = stats[level]["total"]
            stats[level]["accuracy"] = stats[level]["correct"] / total if total > 0 else 0.0

        return stats


class BenchmarkRunner:
    """Runs GAIA benchmark evaluation."""

    def __init__(
        self,
        config: Optional[Config] = None,
        data_dir: Optional[Path] = None,
    ):
        """Initialize benchmark runner.

        Args:
            config: Configuration object
            data_dir: Path to GAIA dataset directory
        """
        self.config = config or get_config()
        self.data_dir = Path(data_dir) if data_dir else self.config.gaia_dir
        self.loader = GAIALoader(self.data_dir)
        self.evaluator = Evaluator()
        self.agent = GAIAAgent(self.config)

    async def run_task(
        self,
        task: GAIATask,
        working_dir: Optional[Path] = None,
    ) -> TaskResult:
        """Run a single benchmark task.

        Args:
            task: GAIA task to run
            working_dir: Optional working directory

        Returns:
            TaskResult with answer and evaluation
        """
        start_time = time.time()

        try:
            # Get file path if task has file
            file_path = None
            if task.has_file():
                file_path = task.get_absolute_file_path(self.data_dir)
                if file_path and not file_path.exists():
                    logger.warning(f"Task file not found: {file_path}")
                    file_path = None

            # Run the agent
            answer, state = await self.agent.solve_with_retries(
                question=task.question,
                file_path=str(file_path) if file_path else None,
                task_id=task.task_id,
                working_dir=str(working_dir) if working_dir else None,
            )

            execution_time = time.time() - start_time

            # Evaluate if we have expected answer
            evaluation = None
            if task.final_answer is not None:
                evaluation = self.evaluator.evaluate(
                    predicted=answer or "",
                    expected=task.final_answer,
                    task_id=task.task_id,
                )

            return TaskResult(
                task=task,
                agent_answer=answer,
                evaluation=evaluation,
                state=state,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {e}")

            return TaskResult(
                task=task,
                agent_answer=None,
                evaluation=None,
                state=None,
                execution_time=execution_time,
                error=str(e),
            )

    async def run_benchmark(
        self,
        split: str = "validation",
        level: Optional[int] = None,
        max_tasks: Optional[int] = None,
        working_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark on multiple tasks.

        Args:
            split: Dataset split
            level: Optional level filter
            max_tasks: Maximum tasks to run
            working_dir: Optional working directory
            show_progress: Whether to show progress bar

        Returns:
            BenchmarkResult with all results
        """
        # Load tasks
        tasks = self.loader.load_tasks(split, level, max_tasks=max_tasks)

        if not tasks:
            console.print("[red]No tasks loaded. Check dataset path.[/red]")
            return BenchmarkResult(
                split=split,
                level=level,
                total_tasks=0,
                completed_tasks=0,
                correct_tasks=0,
                failed_tasks=0,
                accuracy=0.0,
            )

        # Initialize result
        result = BenchmarkResult(
            split=split,
            level=level,
            total_tasks=len(tasks),
            completed_tasks=0,
            correct_tasks=0,
            failed_tasks=0,
            accuracy=0.0,
        )
        result.start_time = datetime.now()

        console.print(f"\n[bold]Running GAIA Benchmark[/bold]")
        console.print(f"Split: {split}, Level: {level or 'all'}, Tasks: {len(tasks)}\n")

        # Run tasks with progress
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task_progress = progress.add_task("Running tasks...", total=len(tasks))

                for task in tasks:
                    progress.update(task_progress, description=f"Task {task.task_id[:8]}...")

                    task_result = await self.run_task(task, working_dir)
                    result.add_result(task_result)

                    # Log result
                    status = (
                        "✓" if task_result.evaluation and task_result.evaluation.is_correct else "✗"
                    )
                    logger.info(f"[{status}] Task {task.task_id}: {task_result.agent_answer}")

                    progress.advance(task_progress)
        else:
            for task in tasks:
                task_result = await self.run_task(task, working_dir)
                result.add_result(task_result)

        result.end_time = datetime.now()

        # Print summary
        self._print_summary(result)

        return result

    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print benchmark summary."""
        console.print("\n[bold]Benchmark Results[/bold]\n")

        # Overall stats
        table = Table(title="Overall Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Tasks", str(result.total_tasks))
        table.add_row("Completed", str(result.completed_tasks))
        table.add_row("Correct", str(result.correct_tasks))
        table.add_row("Failed", str(result.failed_tasks))
        table.add_row("Accuracy", f"{result.accuracy:.1%}")

        if result.start_time and result.end_time:
            duration = (result.end_time - result.start_time).total_seconds()
            table.add_row("Duration", f"{duration:.1f}s")

        console.print(table)

        # Per-level stats
        level_stats = result.get_level_stats()

        level_table = Table(title="Performance by Level")
        level_table.add_column("Level", style="cyan")
        level_table.add_column("Total", justify="right")
        level_table.add_column("Correct", justify="right")
        level_table.add_column("Accuracy", justify="right", style="green")

        for level in [1, 2, 3]:
            stats = level_stats[level]
            if stats["total"] > 0:
                level_table.add_row(
                    f"Level {level}",
                    str(stats["total"]),
                    str(stats["correct"]),
                    f"{stats['accuracy']:.1%}",
                )

        console.print(level_table)


async def run_benchmark(
    split: str = "validation",
    level: Optional[int] = None,
    max_tasks: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> BenchmarkResult:
    """Convenience function to run benchmark.

    Args:
        split: Dataset split
        level: Optional level filter
        max_tasks: Maximum tasks
        data_dir: Path to GAIA dataset

    Returns:
        BenchmarkResult
    """
    runner = BenchmarkRunner(data_dir=Path(data_dir) if data_dir else None)
    return await runner.run_benchmark(split, level, max_tasks)
