"""Tests for agent tools."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


class TestE2BExecutor:
    """Tests for E2B executor tool."""

    @pytest.mark.asyncio
    async def test_execute_python_simple(self):
        """Test simple code execution."""
        from src.tools.e2b_executor import execute_python

        # Mock the Sandbox
        mock_execution = MagicMock()
        mock_execution.logs = MagicMock()
        mock_execution.logs.stdout = "Hello, World!\n"
        mock_execution.logs.stderr = ""
        mock_execution.error = None
        mock_execution.text = None

        with patch("src.tools.e2b_executor.Sandbox") as MockSandbox:
            mock_sandbox = MagicMock()
            mock_sandbox.run_code.return_value = mock_execution
            mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
            mock_sandbox.__exit__ = MagicMock(return_value=False)
            MockSandbox.return_value = mock_sandbox

            result = await execute_python(
                {
                    "code": "print('Hello, World!')",
                    "timeout": 30,
                }
            )

            assert "content" in result
            assert len(result["content"]) == 1
            assert "Hello, World!" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_python_empty_code(self):
        """Test execution with empty code."""
        from src.tools.e2b_executor import execute_python

        result = await execute_python({"code": ""})

        assert "content" in result
        assert "Error" in result["content"][0]["text"]


class TestRAGSystem:
    """Tests for RAG knowledge base."""

    def test_rag_add_and_query(self, tmp_path):
        """Test adding documents and querying."""
        from src.tools.rag import RAGSystem

        rag = RAGSystem(persist_dir=tmp_path / "chroma")

        # Add documents
        rag.add_document("doc1", "Python is a programming language")
        rag.add_document("doc2", "JavaScript is used for web development")

        # Query
        results = rag.query("programming language", top_k=2)

        assert len(results) > 0
        assert results[0].id == "doc1"

    def test_rag_delete(self, tmp_path):
        """Test document deletion."""
        from src.tools.rag import RAGSystem

        rag = RAGSystem(persist_dir=tmp_path / "chroma")

        rag.add_document("doc1", "Test document content")
        assert rag.count() == 1

        rag.delete_document("doc1")
        assert rag.count() == 0


class TestWebSearch:
    """Tests for web search tool."""

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self):
        """Test search with empty query."""
        from src.tools.web_search import web_search

        result = await web_search({"query": ""})

        assert "content" in result
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_fetch_webpage_empty_url(self):
        """Test fetch with empty URL."""
        from src.tools.web_search import fetch_webpage

        result = await fetch_webpage({"url": ""})

        assert "content" in result
        assert "Error" in result["content"][0]["text"]


class TestFileHandler:
    """Tests for file handler tool."""

    def test_read_text_file(self, tmp_path):
        """Test reading text file."""
        from src.tools.file_handler import FileHandler

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        handler = FileHandler()
        result = handler.read(test_file)

        assert result.filename == "test.txt"
        assert result.text_content == "Hello, World!"
        assert result.mime_type == "text/plain"

    def test_read_json_file(self, tmp_path):
        """Test reading JSON file."""
        from src.tools.file_handler import FileHandler

        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        handler = FileHandler()
        result = handler.read(test_file)

        assert result.filename == "test.json"
        assert "key" in result.text_content
        assert "value" in result.text_content

    def test_read_csv_file(self, tmp_path):
        """Test reading CSV file."""
        from src.tools.file_handler import FileHandler

        test_file = tmp_path / "test.csv"
        test_file.write_text("name,value\nalice,100\nbob,200")

        handler = FileHandler()
        result = handler.read(test_file)

        assert result.filename == "test.csv"
        assert "alice" in result.text_content
        assert result.metadata["rows"] == 3

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file."""
        from src.tools.file_handler import FileHandler

        handler = FileHandler()

        with pytest.raises(FileNotFoundError):
            handler.read("/nonexistent/path.txt")
