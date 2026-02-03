"""Main Autonomous Agent for GAIA Benchmark.

Uses Anthropic's Claude Agent SDK with custom tools for:
- Sandboxed code execution (E2B)
- RAG knowledge retrieval
- Web search
- File handling
"""

import logging
from typing import AsyncIterator, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

from .config import Config, get_config
from .planning import TaskPlanner, TaskState, StepResult, StepStatus
from .tools import execute_python, query_knowledge_base, web_search, read_file
from .tools.web_search import fetch_webpage
from .tools.file_handler import list_files

logger = logging.getLogger(__name__)


# System prompt optimized for GAIA benchmark tasks
SYSTEM_PROMPT = """You are an autonomous AI agent designed to solve complex, real-world tasks from the GAIA benchmark. Your goal is to provide accurate, factual answers.

## Core Principles

1. **Accuracy First**: Your answers must be precise and factual. GAIA tasks have single, unambiguous correct answers.

2. **Tool-Based Reasoning**: Use the available tools strategically:
   - `execute_python`: For calculations, data processing, parsing files, any computation
   - `query_knowledge_base`: For retrieving stored knowledge and documents
   - `web_search`: For finding current information, facts, or researching topics
   - `fetch_webpage`: For reading full content from a specific URL
   - `read_file`: For reading attached files (PDF, spreadsheets, images, text)
   - `list_files`: For listing files in a directory

3. **Methodical Approach**:
   - Read any attached files first to understand the full context
   - Break complex tasks into smaller steps
   - Use web search for current or factual information
   - Use code execution for calculations and data processing
   - Verify your reasoning before providing final answers

4. **Answer Format**:
   - Provide ONLY the final answer when you're confident
   - Answers should be as concise as possible
   - For numbers: provide the exact numerical value
   - For names/text: provide the exact string
   - Do not include explanations in your final answer

5. **Error Recovery**:
   - If a tool fails, try alternative approaches
   - If you're stuck, break the problem down differently
   - Always aim for the most reliable path to the answer

## Response Style

When working on a task:
- Think step by step, using tools as needed
- Show your reasoning process
- When ready, clearly state: "FINAL ANSWER: [your answer]"
"""


class GAIAAgent:
    """Autonomous agent for GAIA benchmark tasks."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the GAIA agent.

        Args:
            config: Configuration object, or None to use global config
        """
        self.config = config or get_config()
        self.planner = TaskPlanner()
        self._client: Optional[ClaudeSDKClient] = None
        self._mcp_server = None

    def _create_mcp_server(self):
        """Create the MCP server with all tools."""
        if self._mcp_server is None:
            self._mcp_server = create_sdk_mcp_server(
                name="gaia-tools",
                version="1.0.0",
                tools=[
                    execute_python,
                    query_knowledge_base,
                    web_search,
                    fetch_webpage,
                    read_file,
                    list_files,
                ],
            )
        return self._mcp_server

    def _create_options(self, working_dir: Optional[str] = None) -> ClaudeAgentOptions:
        """Create ClaudeAgentOptions for the agent.

        Args:
            working_dir: Optional working directory path

        Returns:
            Configured ClaudeAgentOptions
        """
        mcp_server = self._create_mcp_server()

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=self.config.model.max_turns,
            mcp_servers={"tools": mcp_server},
            allowed_tools=[
                "mcp__tools__execute_python",
                "mcp__tools__query_knowledge_base",
                "mcp__tools__web_search",
                "mcp__tools__fetch_webpage",
                "mcp__tools__read_file",
                "mcp__tools__list_files",
            ],
            permission_mode="acceptEdits",  # Auto-accept tool use
        )

        if working_dir:
            options.cwd = working_dir

        return options

    async def solve_task(
        self,
        question: str,
        file_path: Optional[str] = None,
        task_id: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> tuple[Optional[str], TaskState]:
        """Solve a GAIA benchmark task.

        Args:
            question: The task question
            file_path: Optional path to attached file
            task_id: Optional task identifier
            working_dir: Optional working directory

        Returns:
            Tuple of (final_answer, task_state)
        """
        # Create task state
        task_id = task_id or f"task_{id(question)}"
        state = TaskState(
            task_id=task_id,
            question=question,
            file_path=file_path,
        )
        state.start()

        # Build the prompt
        prompt = self._build_prompt(question, file_path)

        # Create agent options
        options = self._create_options(working_dir)

        try:
            async with ClaudeSDKClient(options=options) as client:
                # Send the task
                await client.query(prompt)

                # Process responses
                final_answer = None
                full_response = []

                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                text = block.text
                                full_response.append(text)

                                # Check for final answer
                                answer = self._extract_final_answer(text)
                                if answer:
                                    final_answer = answer

                            elif isinstance(block, ToolUseBlock):
                                logger.info(f"Tool use: {block.name}")

                            elif isinstance(block, ToolResultBlock):
                                logger.debug(f"Tool result received")

                # Record result
                state.complete(final_answer or "")
                state.add_step_result(
                    StepResult(
                        step_id="main",
                        status=StepStatus.COMPLETED if final_answer else StepStatus.FAILED,
                        output="\n".join(full_response),
                    )
                )

                return final_answer, state

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            state.fail(str(e))
            state.add_step_result(
                StepResult(
                    step_id="main",
                    status=StepStatus.FAILED,
                    error=str(e),
                )
            )
            return None, state

    def _build_prompt(self, question: str, file_path: Optional[str] = None) -> str:
        """Build the task prompt for the agent.

        Args:
            question: The task question
            file_path: Optional path to attached file

        Returns:
            Complete prompt string
        """
        parts = ["## Task\n"]
        parts.append(question)

        if file_path:
            parts.append(f"\n\n## Attached File\nPath: {file_path}")
            parts.append("Please read this file first to understand the context.")

        # Add planning hints
        has_file = file_path is not None
        plan = self.planner.create_initial_plan(question, has_file)
        plan_prompt = self.planner.get_plan_prompt(plan)

        if plan_prompt:
            parts.append(f"\n\n## Suggested Approach\n{plan_prompt}")

        parts.append("\n\nPlease solve this task and provide your FINAL ANSWER.")

        return "\n".join(parts)

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from agent response.

        Args:
            text: Response text from agent

        Returns:
            Extracted answer or None
        """
        import re

        # Look for explicit final answer markers
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up answer
                answer = answer.strip("\"'").strip()
                if answer:
                    return answer

        return None

    async def solve_with_retries(
        self,
        question: str,
        file_path: Optional[str] = None,
        task_id: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_retries: int = 3,
    ) -> tuple[Optional[str], TaskState]:
        """Solve a task with automatic retries on failure.

        Args:
            question: The task question
            file_path: Optional path to attached file
            task_id: Optional task identifier
            working_dir: Optional working directory
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (final_answer, task_state)
        """
        last_state = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries}")

            answer, state = await self.solve_task(
                question=question,
                file_path=file_path,
                task_id=f"{task_id}_attempt{attempt}" if task_id else None,
                working_dir=working_dir,
            )

            last_state = state

            if answer:
                return answer, state

            # If no answer, wait before retry
            if attempt < max_retries:
                import asyncio

                await asyncio.sleep(1)

        return None, last_state


async def run_single_task(
    question: str,
    file_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> tuple[Optional[str], TaskState]:
    """Convenience function to run a single task.

    Args:
        question: The task question
        file_path: Optional path to attached file
        config: Optional configuration

    Returns:
        Tuple of (final_answer, task_state)
    """
    agent = GAIAAgent(config)
    return await agent.solve_task(question, file_path)
