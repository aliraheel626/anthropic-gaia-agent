"""Answer Evaluator for GAIA Benchmark.

Evaluates agent answers against expected answers using
flexible matching strategies.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating an answer."""

    task_id: str
    predicted: str
    expected: str
    is_correct: bool
    match_type: str = "none"  # exact, normalized, numeric, partial
    confidence: float = 1.0
    details: Optional[str] = None


class Evaluator:
    """Evaluates agent answers against expected answers."""

    def __init__(self, strict: bool = False):
        """Initialize evaluator.

        Args:
            strict: If True, require exact matches only
        """
        self.strict = strict

    def evaluate(
        self,
        predicted: str,
        expected: str,
        task_id: str = "",
    ) -> EvaluationResult:
        """Evaluate a predicted answer against expected.

        Args:
            predicted: Agent's predicted answer
            expected: Expected correct answer
            task_id: Task identifier for logging

        Returns:
            EvaluationResult with match details
        """
        if not predicted or not expected:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted or "",
                expected=expected or "",
                is_correct=False,
                match_type="none",
                details="Missing answer",
            )

        # Try different matching strategies

        # 1. Exact match
        if predicted.strip() == expected.strip():
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=True,
                match_type="exact",
            )

        # 2. Normalized match (case-insensitive, stripped)
        pred_norm = self._normalize(predicted)
        exp_norm = self._normalize(expected)

        if pred_norm == exp_norm:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=True,
                match_type="normalized",
            )

        if self.strict:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=False,
                match_type="none",
            )

        # 3. Numeric match (for numerical answers)
        numeric_match = self._numeric_match(predicted, expected)
        if numeric_match:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=True,
                match_type="numeric",
            )

        # 4. Contains match (predicted contains expected or vice versa)
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=True,
                match_type="partial",
                confidence=0.8,
                details="Partial match",
            )

        # 5. Token overlap (for multi-word answers)
        overlap = self._token_overlap(predicted, expected)
        if overlap >= 0.8:
            return EvaluationResult(
                task_id=task_id,
                predicted=predicted,
                expected=expected,
                is_correct=True,
                match_type="partial",
                confidence=overlap,
                details=f"Token overlap: {overlap:.0%}",
            )

        # No match
        return EvaluationResult(
            task_id=task_id,
            predicted=predicted,
            expected=expected,
            is_correct=False,
            match_type="none",
            details=f"Token overlap: {overlap:.0%}",
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation (except necessary ones)
        text = re.sub(r"[^\w\s\.\-]", "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Strip
        text = text.strip()

        return text

    def _numeric_match(self, predicted: str, expected: str) -> bool:
        """Check if answers match numerically.

        Args:
            predicted: Predicted answer
            expected: Expected answer

        Returns:
            True if numerically equivalent
        """
        try:
            # Extract numbers from both
            pred_nums = re.findall(r"-?\d+\.?\d*", predicted)
            exp_nums = re.findall(r"-?\d+\.?\d*", expected)

            if not pred_nums or not exp_nums:
                return False

            # Compare main numbers
            pred_val = float(pred_nums[0])
            exp_val = float(exp_nums[0])

            # Allow small tolerance for floating point
            if abs(pred_val - exp_val) < 0.001:
                return True

            # Check for percentage/decimal equivalence
            if abs(pred_val - exp_val * 100) < 0.01:
                return True
            if abs(pred_val * 100 - exp_val) < 0.01:
                return True

            return False

        except (ValueError, IndexError):
            return False

    def _token_overlap(self, predicted: str, expected: str) -> float:
        """Calculate token overlap between answers.

        Args:
            predicted: Predicted answer
            expected: Expected answer

        Returns:
            Overlap ratio (0-1)
        """
        pred_tokens = set(self._normalize(predicted).split())
        exp_tokens = set(self._normalize(expected).split())

        if not exp_tokens:
            return 0.0

        intersection = pred_tokens & exp_tokens
        return len(intersection) / len(exp_tokens)

    def batch_evaluate(
        self,
        predictions: list[tuple[str, str, str]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of predictions.

        Args:
            predictions: List of (task_id, predicted, expected) tuples

        Returns:
            List of EvaluationResult
        """
        return [self.evaluate(pred, exp, task_id) for task_id, pred, exp in predictions]

    def compute_metrics(self, results: list[EvaluationResult]) -> dict:
        """Compute aggregate metrics from results.

        Args:
            results: List of EvaluationResult

        Returns:
            Dictionary with metrics
        """
        if not results:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        correct = sum(1 for r in results if r.is_correct)
        total = len(results)

        return {
            "accuracy": correct / total,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "by_match_type": self._count_by_match_type(results),
        }

    def _count_by_match_type(self, results: list[EvaluationResult]) -> dict:
        """Count results by match type."""
        counts = {}
        for r in results:
            counts[r.match_type] = counts.get(r.match_type, 0) + 1
        return counts
