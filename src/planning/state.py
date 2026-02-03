"""Task State Management.

Manages task-level state across multi-step agent operations.
Enables recovery from failures and tracks intermediate results.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a task step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorType(Enum):
    """Classification of errors for recovery decisions."""

    REASONING = "reasoning"  # Model made wrong inference
    EXECUTION = "execution"  # Code execution failed
    RETRIEVAL = "retrieval"  # Knowledge/search retrieval failed
    TIMEOUT = "timeout"  # Operation timed out
    VALIDATION = "validation"  # Answer validation failed
    UNKNOWN = "unknown"


@dataclass
class StepResult:
    """Result of executing a task step."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    execution_time: float = 0.0
    tool_calls: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output if not callable(self.output) else str(self.output),
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "execution_time": self.execution_time,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepResult":
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            status=StepStatus(data["status"]),
            output=data.get("output"),
            error=data.get("error"),
            error_type=ErrorType(data["error_type"]) if data.get("error_type") else None,
            execution_time=data.get("execution_time", 0.0),
            tool_calls=data.get("tool_calls", []),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
        )


@dataclass
class TaskState:
    """Complete state for a task execution."""

    task_id: str
    question: str
    file_path: Optional[str] = None
    expected_answer: Optional[str] = None  # For validation (only in dev set)

    # Execution state
    current_step: int = 0
    steps: list = field(default_factory=list)
    results: list = field(default_factory=list)

    # Meta information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    final_answer: Optional[str] = None
    total_tokens: int = 0

    # Error tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None

    def start(self) -> None:
        """Mark task as started."""
        self.start_time = datetime.now()

    def complete(self, answer: str) -> None:
        """Mark task as completed with final answer."""
        self.end_time = datetime.now()
        self.final_answer = answer

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.end_time = datetime.now()
        self.last_error = error

    def add_step_result(self, result: StepResult) -> None:
        """Add a step result to the state."""
        self.results.append(result)

        if result.status == StepStatus.COMPLETED:
            self.current_step += 1

    def get_last_result(self) -> Optional[StepResult]:
        """Get the most recent step result."""
        return self.results[-1] if self.results else None

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1

    def get_context_summary(self) -> str:
        """Get a summary of current state for context injection.

        Returns:
            String summary of completed steps and results
        """
        if not self.results:
            return ""

        summary_parts = ["Previous steps:"]

        for i, result in enumerate(self.results):
            status_icon = "✓" if result.status == StepStatus.COMPLETED else "✗"
            summary_parts.append(f"{i + 1}. [{status_icon}] {result.step_id}")

            if result.output and result.status == StepStatus.COMPLETED:
                output_str = str(result.output)[:200]
                if len(str(result.output)) > 200:
                    output_str += "..."
                summary_parts.append(f"   Result: {output_str}")

        return "\n".join(summary_parts)

    def get_elapsed_time(self) -> float:
        """Get total elapsed time in seconds."""
        if not self.start_time:
            return 0.0

        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.final_answer is not None or self.end_time is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "file_path": self.file_path,
            "expected_answer": self.expected_answer,
            "current_step": self.current_step,
            "steps": self.steps,
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """Create from dictionary."""
        state = cls(
            task_id=data["task_id"],
            question=data["question"],
            file_path=data.get("file_path"),
            expected_answer=data.get("expected_answer"),
            current_step=data.get("current_step", 0),
            steps=data.get("steps", []),
            total_tokens=data.get("total_tokens", 0),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_error=data.get("last_error"),
            final_answer=data.get("final_answer"),
        )

        state.results = [StepResult.from_dict(r) for r in data.get("results", [])]

        if data.get("start_time"):
            state.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            state.end_time = datetime.fromisoformat(data["end_time"])

        return state

    def save(self, path: Path) -> None:
        """Save state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TaskState":
        """Load state from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class StateManager:
    """Manages task states for persistence and recovery."""

    def __init__(self, state_dir: Path):
        """Initialize state manager.

        Args:
            state_dir: Directory for state persistence
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._states: dict[str, TaskState] = {}

    def get_state_path(self, task_id: str) -> Path:
        """Get path for a task's state file."""
        return self.state_dir / f"{task_id}.json"

    def create_state(
        self,
        task_id: str,
        question: str,
        file_path: Optional[str] = None,
        expected_answer: Optional[str] = None,
    ) -> TaskState:
        """Create a new task state."""
        state = TaskState(
            task_id=task_id,
            question=question,
            file_path=file_path,
            expected_answer=expected_answer,
        )
        self._states[task_id] = state
        return state

    def get_state(self, task_id: str) -> Optional[TaskState]:
        """Get a task state, loading from disk if needed."""
        if task_id in self._states:
            return self._states[task_id]

        state_path = self.get_state_path(task_id)
        if state_path.exists():
            state = TaskState.load(state_path)
            self._states[task_id] = state
            return state

        return None

    def save_state(self, task_id: str) -> None:
        """Save a task state to disk."""
        if task_id in self._states:
            state = self._states[task_id]
            state.save(self.get_state_path(task_id))

    def save_all(self) -> None:
        """Save all task states to disk."""
        for task_id in self._states:
            self.save_state(task_id)

    def list_incomplete(self) -> list[str]:
        """List task IDs with incomplete states."""
        incomplete = []

        for path in self.state_dir.glob("*.json"):
            try:
                state = TaskState.load(path)
                if not state.is_complete():
                    incomplete.append(state.task_id)
            except Exception as e:
                logger.warning(f"Error loading state {path}: {e}")

        return incomplete
