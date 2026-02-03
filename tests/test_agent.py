"""Tests for main agent."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestGAIAAgent:
    """Tests for GAIA agent."""

    def test_agent_initialization(self):
        """Test agent can be initialized."""
        from src.agent import GAIAAgent
        from src.config import Config

        config = Config(
            anthropic_api_key="test_key",
            e2b_api_key="test_e2b_key",
        )

        agent = GAIAAgent(config)

        assert agent.config == config
        assert agent.planner is not None

    def test_build_prompt_simple(self):
        """Test prompt building without file."""
        from src.agent import GAIAAgent
        from src.config import Config

        config = Config(
            anthropic_api_key="test_key",
            e2b_api_key="test_e2b_key",
        )
        agent = GAIAAgent(config)

        prompt = agent._build_prompt("What is 2 + 2?")

        assert "What is 2 + 2?" in prompt
        assert "FINAL ANSWER" in prompt

    def test_build_prompt_with_file(self):
        """Test prompt building with file."""
        from src.agent import GAIAAgent
        from src.config import Config

        config = Config(
            anthropic_api_key="test_key",
            e2b_api_key="test_e2b_key",
        )
        agent = GAIAAgent(config)

        prompt = agent._build_prompt("Analyze this file", file_path="/path/to/file.pdf")

        assert "Analyze this file" in prompt
        assert "/path/to/file.pdf" in prompt
        assert "Attached File" in prompt

    def test_extract_final_answer(self):
        """Test answer extraction from response."""
        from src.agent import GAIAAgent
        from src.config import Config

        config = Config(
            anthropic_api_key="test_key",
            e2b_api_key="test_e2b_key",
        )
        agent = GAIAAgent(config)

        # Test various answer formats
        text1 = "After analysis, FINAL ANSWER: 42"
        assert agent._extract_final_answer(text1) == "42"

        text2 = "The answer is: Paris"
        assert agent._extract_final_answer(text2) == "Paris"

        text3 = "Let me think about this..."
        assert agent._extract_final_answer(text3) is None


class TestTaskPlanner:
    """Tests for task planner."""

    def test_analyze_question_code(self):
        """Test code-related question analysis."""
        from src.planning import TaskPlanner
        from src.planning.planner import ToolType

        planner = TaskPlanner()

        tools = planner.analyze_question("Calculate the sum of numbers in the file")

        assert ToolType.CODE_EXECUTION in tools

    def test_analyze_question_search(self):
        """Test search-related question analysis."""
        from src.planning import TaskPlanner
        from src.planning.planner import ToolType

        planner = TaskPlanner()

        tools = planner.analyze_question("What is the current population of Tokyo?")

        assert ToolType.WEB_SEARCH in tools

    def test_analyze_question_file(self):
        """Test file-related question analysis."""
        from src.planning import TaskPlanner
        from src.planning.planner import ToolType

        planner = TaskPlanner()

        tools = planner.analyze_question("Read the pdf and summarize")

        assert ToolType.READ_FILE in tools

    def test_create_plan_with_file(self):
        """Test plan creation with file."""
        from src.planning import TaskPlanner
        from src.planning.planner import ToolType

        planner = TaskPlanner()

        steps = planner.create_initial_plan("Analyze the data", has_file=True)

        # First step should be file reading
        assert len(steps) > 0
        assert steps[0].tool_hint == ToolType.READ_FILE

    def test_complexity_estimation(self):
        """Test complexity estimation."""
        from src.planning import TaskPlanner

        planner = TaskPlanner()

        # Simple question
        simple = planner.classify_question_complexity("What is 2+2?")
        assert simple == 1

        # Complex multi-step
        complex_q = planner.classify_question_complexity(
            "First, search for the data, then calculate the sum, and finally report the result"
        )
        assert complex_q >= 2


class TestTaskState:
    """Tests for task state management."""

    def test_state_lifecycle(self):
        """Test state lifecycle."""
        from src.planning import TaskState

        state = TaskState(
            task_id="test_001",
            question="What is the answer?",
        )

        assert not state.is_complete()

        state.start()
        assert state.start_time is not None

        state.complete("42")
        assert state.is_complete()
        assert state.final_answer == "42"

    def test_state_serialization(self, tmp_path):
        """Test state save and load."""
        from src.planning import TaskState

        state = TaskState(
            task_id="test_002",
            question="Test question",
        )
        state.start()
        state.complete("answer")

        # Save
        save_path = tmp_path / "state.json"
        state.save(save_path)

        # Load
        loaded = TaskState.load(save_path)

        assert loaded.task_id == state.task_id
        assert loaded.question == state.question
        assert loaded.final_answer == state.final_answer
