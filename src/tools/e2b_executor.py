"""E2B Sandbox Executor Tool.

Provides secure, sandboxed Python code execution using E2B infrastructure.
All code execution happens in isolated cloud sandboxes, never locally.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from claude_agent_sdk import tool
from e2b_code_interpreter import Sandbox

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


class E2BExecutor:
    """Manages E2B sandbox sessions for code execution."""

    def __init__(self, timeout: int = 300, memory_mb: int = 2048):
        """Initialize executor with sandbox settings.

        Args:
            timeout: Maximum execution time in seconds
            memory_mb: Memory allocation for sandbox
        """
        self.timeout = timeout
        self.memory_mb = memory_mb
        self._sandbox: Optional[Sandbox] = None

    def __enter__(self) -> "E2BExecutor":
        """Create sandbox on context entry."""
        self._sandbox = Sandbox()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up sandbox on context exit."""
        if self._sandbox:
            try:
                self._sandbox.kill()
            except Exception as e:
                logger.warning(f"Error killing sandbox: {e}")
            self._sandbox = None

    def execute(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            timeout: Optional timeout override

        Returns:
            ExecutionResult with stdout, stderr, and any return value
        """
        if not self._sandbox:
            # Create temporary sandbox if not in context
            with Sandbox() as sbx:
                return self._run_code(sbx, code, timeout or self.timeout)

        return self._run_code(self._sandbox, code, timeout or self.timeout)

    def _run_code(self, sandbox: Sandbox, code: str, timeout: int) -> ExecutionResult:
        """Internal method to run code in a sandbox."""
        import time

        start_time = time.time()

        try:
            execution = sandbox.run_code(code, timeout=timeout)
            elapsed = time.time() - start_time

            # Extract logs
            stdout = ""
            stderr = ""

            if execution.logs:
                stdout = (
                    execution.logs.stdout
                    if hasattr(execution.logs, "stdout")
                    else str(execution.logs)
                )
                stderr = execution.logs.stderr if hasattr(execution.logs, "stderr") else ""

            # Check for errors
            if execution.error:
                return ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=str(execution.error),
                    execution_time=elapsed,
                )

            return ExecutionResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                result=execution.text if hasattr(execution, "text") else None,
                execution_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Sandbox execution error: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=elapsed,
            )

    def upload_file(self, local_path: str, sandbox_path: str) -> bool:
        """Upload a file to the sandbox.

        Args:
            local_path: Path to local file
            sandbox_path: Destination path in sandbox

        Returns:
            True if successful
        """
        if not self._sandbox:
            logger.error("No active sandbox for file upload")
            return False

        try:
            with open(local_path, "rb") as f:
                content = f.read()
            self._sandbox.files.write(sandbox_path, content)
            return True
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return False

    def download_file(self, sandbox_path: str) -> Optional[bytes]:
        """Download a file from the sandbox.

        Args:
            sandbox_path: Path in sandbox

        Returns:
            File contents as bytes, or None on error
        """
        if not self._sandbox:
            logger.error("No active sandbox for file download")
            return None

        try:
            return self._sandbox.files.read(sandbox_path)
        except Exception as e:
            logger.error(f"File download error: {e}")
            return None


# Tool function for Claude Agent SDK
@tool(
    "execute_python",
    "Execute Python code in a secure E2B sandbox. Use this for calculations, data processing, or any code execution. Returns stdout, stderr, and any return value.",
    {
        "code": str,
        "timeout": int,  # Optional timeout in seconds
    },
)
async def execute_python(args: dict) -> dict:
    """Execute Python code in E2B sandbox.

    Args:
        args: Dictionary with 'code' (required) and 'timeout' (optional)

    Returns:
        Dictionary with execution results
    """
    code = args.get("code", "")
    timeout = args.get("timeout", 60)

    if not code.strip():
        return {"content": [{"type": "text", "text": "Error: No code provided to execute."}]}

    config = get_config()

    try:
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code, timeout=timeout)

            # Build response
            response_parts = []

            if execution.logs:
                stdout = execution.logs.stdout if hasattr(execution.logs, "stdout") else ""
                stderr = execution.logs.stderr if hasattr(execution.logs, "stderr") else ""

                if stdout:
                    response_parts.append(f"**stdout:**\n```\n{stdout}\n```")
                if stderr:
                    response_parts.append(f"**stderr:**\n```\n{stderr}\n```")

            if execution.error:
                response_parts.append(f"**Error:**\n```\n{execution.error}\n```")

            if hasattr(execution, "text") and execution.text:
                response_parts.append(f"**Result:** {execution.text}")

            if not response_parts:
                response_parts.append("Code executed successfully with no output.")

            return {"content": [{"type": "text", "text": "\n\n".join(response_parts)}]}

    except Exception as e:
        logger.error(f"E2B execution error: {e}")
        return {"content": [{"type": "text", "text": f"Execution error: {str(e)}"}]}
