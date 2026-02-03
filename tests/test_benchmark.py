"""Tests for benchmark module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestEvaluator:
    """Tests for answer evaluator."""

    def test_exact_match(self):
        """Test exact string match."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()
        result = evaluator.evaluate("Paris", "Paris", "task_1")

        assert result.is_correct
        assert result.match_type == "exact"

    def test_normalized_match(self):
        """Test case-insensitive match."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()
        result = evaluator.evaluate("paris", "Paris", "task_1")

        assert result.is_correct
        assert result.match_type == "normalized"

    def test_numeric_match(self):
        """Test numeric equivalence."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()

        # Integer match
        result = evaluator.evaluate("42", "42", "task_1")
        assert result.is_correct

        # Float match
        result = evaluator.evaluate("3.14", "3.14", "task_1")
        assert result.is_correct

        # Extracted numeric
        result = evaluator.evaluate("The answer is 42", "42", "task_1")
        assert result.is_correct

    def test_partial_match(self):
        """Test partial/contains match."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()
        result = evaluator.evaluate("New York City", "New York", "task_1")

        assert result.is_correct
        assert result.match_type == "partial"

    def test_no_match(self):
        """Test non-matching answers."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()
        result = evaluator.evaluate("London", "Paris", "task_1")

        assert not result.is_correct
        assert result.match_type == "none"

    def test_batch_evaluate(self):
        """Test batch evaluation."""
        from src.benchmark import Evaluator

        evaluator = Evaluator()
        predictions = [
            ("task_1", "Paris", "Paris"),
            ("task_2", "London", "Paris"),
            ("task_3", "42", "42"),
        ]

        results = evaluator.batch_evaluate(predictions)

        assert len(results) == 3
        assert results[0].is_correct
        assert not results[1].is_correct
        assert results[2].is_correct

    def test_compute_metrics(self):
        """Test metric computation."""
        from src.benchmark import Evaluator, EvaluationResult

        evaluator = Evaluator()

        results = [
            EvaluationResult("t1", "a", "a", True, "exact"),
            EvaluationResult("t2", "b", "c", False, "none"),
            EvaluationResult("t3", "d", "d", True, "exact"),
        ]

        metrics = evaluator.compute_metrics(results)

        assert metrics["total"] == 3
        assert metrics["correct"] == 2
        assert metrics["accuracy"] == 2 / 3


class TestGAIALoader:
    """Tests for GAIA loader."""

    def test_loader_initialization(self, tmp_path):
        """Test loader can be initialized."""
        from src.benchmark import GAIALoader

        loader = GAIALoader(tmp_path)
        assert loader.data_dir == tmp_path

    def test_load_jsonl(self, tmp_path):
        """Test loading from JSONL file."""
        from src.benchmark import GAIALoader
        import json

        # Create test JSONL
        (tmp_path / "2023" / "validation").mkdir(parents=True)
        jsonl_path = tmp_path / "2023" / "validation" / "metadata.jsonl"

        tasks = [
            {"task_id": "1", "Question": "What is 2+2?", "Level": 1, "Final answer": "4"},
            {"task_id": "2", "Question": "Capital of France?", "Level": 1, "Final answer": "Paris"},
        ]

        with open(jsonl_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        loader = GAIALoader(tmp_path)
        loaded = loader.load_tasks("validation", year="2023")

        assert len(loaded) == 2
        assert loaded[0].task_id == "1"
        assert loaded[0].question == "What is 2+2?"
        assert loaded[0].final_answer == "4"

    def test_filter_by_level(self, tmp_path):
        """Test filtering by difficulty level."""
        from src.benchmark import GAIALoader
        import json

        # Create test JSONL with mixed levels
        (tmp_path / "2023" / "validation").mkdir(parents=True)
        jsonl_path = tmp_path / "2023" / "validation" / "metadata.jsonl"

        tasks = [
            {"task_id": "1", "Question": "Easy question", "Level": 1},
            {"task_id": "2", "Question": "Medium question", "Level": 2},
            {"task_id": "3", "Question": "Hard question", "Level": 3},
        ]

        with open(jsonl_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        loader = GAIALoader(tmp_path)

        level1 = loader.load_tasks("validation", level=1, year="2023")
        assert len(level1) == 1
        assert level1[0].level == 1

        level2 = loader.load_tasks("validation", level=2, year="2023")
        assert len(level2) == 1
        assert level2[0].level == 2


class TestBenchmarkResult:
    """Tests for benchmark result."""

    def test_add_result(self):
        """Test adding task results."""
        from src.benchmark.runner import BenchmarkResult, TaskResult
        from src.benchmark import GAIATask, EvaluationResult

        result = BenchmarkResult(
            split="validation",
            level=1,
            total_tasks=3,
            completed_tasks=0,
            correct_tasks=0,
            failed_tasks=0,
            accuracy=0.0,
        )

        task = GAIATask("1", "Question", 1)
        eval_result = EvaluationResult("1", "answer", "answer", True, "exact")

        result.add_result(
            TaskResult(
                task=task,
                agent_answer="answer",
                evaluation=eval_result,
                state=None,
                execution_time=1.0,
            )
        )

        assert result.completed_tasks == 1
        assert result.correct_tasks == 1
        assert result.accuracy == 1.0

    def test_level_stats(self):
        """Test per-level statistics."""
        from src.benchmark.runner import BenchmarkResult, TaskResult
        from src.benchmark import GAIATask, EvaluationResult

        result = BenchmarkResult(
            split="validation",
            level=None,
            total_tasks=3,
            completed_tasks=0,
            correct_tasks=0,
            failed_tasks=0,
            accuracy=0.0,
        )

        # Add results from different levels
        for level, correct in [(1, True), (1, False), (2, True)]:
            task = GAIATask(f"task_{level}", "Q", level)
            eval_result = EvaluationResult(
                task.task_id, "a", "a" if correct else "b", correct, "exact"
            )
            result.add_result(
                TaskResult(task=task, agent_answer="a", evaluation=eval_result, state=None)
            )

        stats = result.get_level_stats()

        assert stats[1]["total"] == 2
        assert stats[1]["correct"] == 1
        assert stats[2]["total"] == 1
        assert stats[2]["correct"] == 1
