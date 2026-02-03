"""Task Planner for GAIA Agent.

Handles task decomposition and planning for complex multi-step tasks.
Determines optimal tool selection and step ordering.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools available to the agent."""

    CODE_EXECUTION = "execute_python"
    KNOWLEDGE_BASE = "query_knowledge_base"
    WEB_SEARCH = "web_search"
    FETCH_WEBPAGE = "fetch_webpage"
    READ_FILE = "read_file"
    LIST_FILES = "list_files"


@dataclass
class TaskStep:
    """A single step in a task plan."""

    id: str
    description: str
    tool_hint: Optional[ToolType] = None
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False

    def __post_init__(self):
        if isinstance(self.tool_hint, str):
            try:
                self.tool_hint = ToolType(self.tool_hint)
            except ValueError:
                self.tool_hint = None


class TaskPlanner:
    """Plans task execution by analyzing questions and determining steps.

    The planner provides hints to the agent about which tools might be useful,
    but the agent makes final decisions about tool usage.
    """

    # Keywords that suggest certain tool usage
    TOOL_KEYWORDS = {
        ToolType.CODE_EXECUTION: [
            "calculate",
            "compute",
            "sum",
            "count",
            "average",
            "total",
            "parse",
            "extract",
            "process",
            "analyze",
            "convert",
            "python",
            "code",
            "script",
            "algorithm",
            "formula",
        ],
        ToolType.WEB_SEARCH: [
            "current",
            "latest",
            "recent",
            "today",
            "now",
            "who is",
            "what is",
            "when did",
            "where is",
            "find",
            "search",
            "look up",
            "discover",
        ],
        ToolType.KNOWLEDGE_BASE: [
            "documentation",
            "manual",
            "reference",
            "guide",
            "according to",
            "based on",
            "in the",
        ],
        ToolType.READ_FILE: [
            "file",
            "document",
            "pdf",
            "spreadsheet",
            "excel",
            "csv",
            "image",
            "attachment",
            "given",
        ],
    }

    def __init__(self):
        """Initialize task planner."""
        pass

    def analyze_question(self, question: str, has_file: bool = False) -> list[ToolType]:
        """Analyze a question to suggest potentially useful tools.

        Args:
            question: The task question
            has_file: Whether the task has an associated file

        Returns:
            List of suggested tool types in order of likely usefulness
        """
        question_lower = question.lower()
        tool_scores: dict[ToolType, int] = {t: 0 for t in ToolType}

        # Score each tool based on keyword matches
        for tool, keywords in self.TOOL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    tool_scores[tool] += 1

        # Boost file reading if task has a file
        if has_file:
            tool_scores[ToolType.READ_FILE] += 5

        # Sort by score, filter zeros
        suggested = sorted(
            [(t, s) for t, s in tool_scores.items() if s > 0], key=lambda x: x[1], reverse=True
        )

        return [t for t, _ in suggested]

    def create_initial_plan(
        self,
        question: str,
        has_file: bool = False,
    ) -> list[TaskStep]:
        """Create an initial plan for a task.

        This is a simple heuristic-based planner. The agent will refine
        the plan based on actual execution.

        Args:
            question: The task question
            has_file: Whether the task has an associated file

        Returns:
            List of TaskStep objects
        """
        steps = []
        step_id = 0

        # Step 1: Read file if present
        if has_file:
            steps.append(
                TaskStep(
                    id=f"step_{step_id}",
                    description="Read and understand the attached file",
                    tool_hint=ToolType.READ_FILE,
                )
            )
            step_id += 1

        # Step 2: Analyze which tools might be needed
        suggested_tools = self.analyze_question(question, has_file)

        # Step 3: Add research step if web search suggested
        if ToolType.WEB_SEARCH in suggested_tools:
            steps.append(
                TaskStep(
                    id=f"step_{step_id}",
                    description="Search for relevant information",
                    tool_hint=ToolType.WEB_SEARCH,
                    depends_on=[s.id for s in steps],
                )
            )
            step_id += 1

        # Step 4: Add knowledge base step if suggested
        if ToolType.KNOWLEDGE_BASE in suggested_tools:
            steps.append(
                TaskStep(
                    id=f"step_{step_id}",
                    description="Query knowledge base for relevant information",
                    tool_hint=ToolType.KNOWLEDGE_BASE,
                    depends_on=[s.id for s in steps],
                    optional=True,
                )
            )
            step_id += 1

        # Step 5: Add computation step if needed
        if ToolType.CODE_EXECUTION in suggested_tools:
            steps.append(
                TaskStep(
                    id=f"step_{step_id}",
                    description="Execute code to compute or process data",
                    tool_hint=ToolType.CODE_EXECUTION,
                    depends_on=[s.id for s in steps],
                )
            )
            step_id += 1

        # Step 6: Final reasoning step
        steps.append(
            TaskStep(
                id=f"step_{step_id}",
                description="Synthesize findings and formulate final answer",
                tool_hint=None,  # Pure reasoning, no tool
                depends_on=[s.id for s in steps],
            )
        )

        return steps

    def get_plan_prompt(self, steps: list[TaskStep]) -> str:
        """Generate a prompt describing the plan.

        Args:
            steps: List of TaskStep objects

        Returns:
            String prompt for the agent
        """
        if not steps:
            return ""

        prompt_parts = ["Here is a suggested approach:"]

        for i, step in enumerate(steps, 1):
            tool_hint = f" (consider using {step.tool_hint.value})" if step.tool_hint else ""
            optional = " [optional]" if step.optional else ""
            prompt_parts.append(f"{i}. {step.description}{tool_hint}{optional}")

        prompt_parts.append("\nAdapt this plan as needed based on what you discover.")

        return "\n".join(prompt_parts)

    def classify_question_complexity(self, question: str, has_file: bool = False) -> int:
        """Estimate the complexity level of a question (1-3, matching GAIA levels).

        Args:
            question: The task question
            has_file: Whether the task has an associated file

        Returns:
            Estimated complexity level (1, 2, or 3)
        """
        # Count potential steps needed
        suggested_tools = self.analyze_question(question, has_file)
        num_tools = len(suggested_tools)

        # Check for multi-hop indicators
        multi_hop_keywords = [
            "then",
            "after",
            "next",
            "finally",
            "first",
            "second",
            "step by step",
            "multiple",
            "several",
            "all",
        ]
        question_lower = question.lower()
        multi_hop_score = sum(1 for k in multi_hop_keywords if k in question_lower)

        # Estimate complexity
        if num_tools <= 1 and multi_hop_score == 0:
            return 1  # Simple, minimal tool use
        elif num_tools <= 2 and multi_hop_score <= 2:
            return 2  # Moderate complexity
        else:
            return 3  # Complex, multi-step reasoning
