"""GAIA Dataset Loader.

Loads GAIA benchmark tasks from the HuggingFace dataset.
Supports both validation and test sets at all difficulty levels.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class GAIATask:
    """A single GAIA benchmark task."""

    task_id: str
    question: str
    level: int  # 1, 2, or 3
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    final_answer: Optional[str] = None  # Only in validation set
    annotator_metadata: Optional[dict] = None

    def has_file(self) -> bool:
        """Check if task has an associated file."""
        return self.file_name is not None and self.file_name != ""

    def get_absolute_file_path(self, data_dir: Path) -> Optional[Path]:
        """Get absolute path to the task's file.

        Args:
            data_dir: Base directory of GAIA dataset

        Returns:
            Absolute path or None
        """
        if not self.file_path:
            return None

        return data_dir / self.file_path


class GAIALoader:
    """Loader for GAIA benchmark dataset."""

    SPLITS = ["validation", "test"]
    LEVELS = [1, 2, 3]

    def __init__(self, data_dir: Path):
        """Initialize GAIA loader.

        Args:
            data_dir: Path to downloaded GAIA dataset directory
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, list[GAIATask]] = {}

    def _find_metadata_file(self, year: str = "2023", split: str = "validation") -> Optional[Path]:
        """Find the metadata file for a split.

        Args:
            year: Dataset year (2023 or 2024)
            split: Split name (validation or test)

        Returns:
            Path to metadata file or None
        """
        # Try different possible locations
        possible_paths = [
            self.data_dir / year / split / "metadata.parquet",
            self.data_dir / year / split / "metadata.jsonl",
            self.data_dir / f"{year}_{split}" / "metadata.parquet",
            self.data_dir / "data" / f"{year}_{split}_metadata.parquet",
            self.data_dir / f"metadata.{split}.parquet",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def load_tasks(
        self,
        split: str = "validation",
        level: Optional[int] = None,
        year: str = "2023",
        max_tasks: Optional[int] = None,
    ) -> list[GAIATask]:
        """Load GAIA tasks from dataset.

        Args:
            split: Dataset split (validation or test)
            level: Optional filter by difficulty level (1, 2, or 3)
            year: Dataset year
            max_tasks: Optional limit on number of tasks

        Returns:
            List of GAIATask objects
        """
        cache_key = f"{year}_{split}_{level}"

        if cache_key in self._cache:
            tasks = self._cache[cache_key]
            return tasks[:max_tasks] if max_tasks else tasks

        metadata_path = self._find_metadata_file(year, split)

        if not metadata_path:
            logger.warning(f"Metadata file not found in {self.data_dir}")
            return self._try_load_from_huggingface(split, level, year, max_tasks)

        tasks = self._load_from_file(metadata_path, level)
        self._cache[cache_key] = tasks

        return tasks[:max_tasks] if max_tasks else tasks

    def _load_from_file(self, path: Path, level: Optional[int] = None) -> list[GAIATask]:
        """Load tasks from a metadata file.

        Args:
            path: Path to metadata file
            level: Optional level filter

        Returns:
            List of GAIATask objects
        """
        tasks = []

        if path.suffix == ".parquet":
            tasks = self._load_parquet(path)
        elif path.suffix == ".jsonl":
            tasks = self._load_jsonl(path)
        else:
            logger.error(f"Unsupported file format: {path.suffix}")
            return []

        # Filter by level if specified
        if level is not None:
            tasks = [t for t in tasks if t.level == level]

        return tasks

    def _load_parquet(self, path: Path) -> list[GAIATask]:
        """Load tasks from Parquet file."""
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            tasks = []

            for _, row in df.iterrows():
                task = GAIATask(
                    task_id=str(row.get("task_id", row.name)),
                    question=str(row.get("Question", "")),
                    level=int(row.get("Level", 1)),
                    file_name=row.get("file_name"),
                    file_path=row.get("file_path"),
                    final_answer=row.get("Final answer"),
                    annotator_metadata=row.get("Annotator Metadata"),
                )
                tasks.append(task)

            return tasks

        except ImportError:
            logger.error("pandas not installed for parquet support")
            return []
        except Exception as e:
            logger.error(f"Error loading parquet: {e}")
            return []

    def _load_jsonl(self, path: Path) -> list[GAIATask]:
        """Load tasks from JSONL file."""
        tasks = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task = GAIATask(
                        task_id=str(data.get("task_id", "")),
                        question=str(data.get("Question", "")),
                        level=int(data.get("Level", 1)),
                        file_name=data.get("file_name"),
                        file_path=data.get("file_path"),
                        final_answer=data.get("Final answer"),
                        annotator_metadata=data.get("Annotator Metadata"),
                    )
                    tasks.append(task)
                except json.JSONDecodeError:
                    continue

        return tasks

    def _try_load_from_huggingface(
        self,
        split: str,
        level: Optional[int],
        year: str,
        max_tasks: Optional[int],
    ) -> list[GAIATask]:
        """Try to load directly from HuggingFace datasets library.

        Args:
            split: Dataset split
            level: Optional level filter
            year: Dataset year
            max_tasks: Maximum tasks to load

        Returns:
            List of GAIATask objects
        """
        try:
            from datasets import load_dataset

            # Construct config name
            config_name = f"{year}_level{level}" if level else f"{year}_all"

            logger.info(f"Loading from HuggingFace: gaia-benchmark/GAIA, config={config_name}")

            dataset = load_dataset(
                "gaia-benchmark/GAIA",
                config_name,
                split=split,
                trust_remote_code=True,
            )

            tasks = []
            for i, example in enumerate(dataset):
                if max_tasks and i >= max_tasks:
                    break

                task = GAIATask(
                    task_id=str(example.get("task_id", f"task_{i}")),
                    question=str(example.get("Question", "")),
                    level=int(example.get("Level", level or 1)),
                    file_name=example.get("file_name"),
                    file_path=example.get("file_path"),
                    final_answer=example.get("Final answer"),
                    annotator_metadata=example.get("Annotator Metadata"),
                )
                tasks.append(task)

            return tasks

        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            return []

    def iterate_tasks(
        self,
        split: str = "validation",
        level: Optional[int] = None,
        year: str = "2023",
    ) -> Iterator[GAIATask]:
        """Iterate over GAIA tasks.

        Args:
            split: Dataset split
            level: Optional level filter
            year: Dataset year

        Yields:
            GAIATask objects
        """
        tasks = self.load_tasks(split, level, year)
        yield from tasks

    def get_task(self, task_id: str) -> Optional[GAIATask]:
        """Get a specific task by ID.

        Args:
            task_id: Task identifier

        Returns:
            GAIATask or None
        """
        # Search all cached splits
        for tasks in self._cache.values():
            for task in tasks:
                if task.task_id == task_id:
                    return task

        # Try loading all tasks to find it
        all_tasks = self.load_tasks("validation") + self.load_tasks("test")
        for task in all_tasks:
            if task.task_id == task_id:
                return task

        return None

    def get_stats(self, split: str = "validation", year: str = "2023") -> dict:
        """Get statistics about the dataset.

        Args:
            split: Dataset split
            year: Dataset year

        Returns:
            Dictionary with task counts by level
        """
        tasks = self.load_tasks(split, year=year)

        stats = {
            "total": len(tasks),
            "level_1": sum(1 for t in tasks if t.level == 1),
            "level_2": sum(1 for t in tasks if t.level == 2),
            "level_3": sum(1 for t in tasks if t.level == 3),
            "with_files": sum(1 for t in tasks if t.has_file()),
        }

        return stats
