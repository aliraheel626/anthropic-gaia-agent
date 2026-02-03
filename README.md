# GAIA Agent

Autonomous AI Agent for GAIA Benchmark using Anthropic Agent SDK.

## Target Performance

- **Current Baseline**: 37-47% accuracy
- **Target**: 55-63% accuracy on GAIA benchmark

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync --all-extras

# Or install with dev dependencies
uv sync --dev
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Download GAIA Dataset (Manual)

```bash
# Install huggingface_hub CLI (if not already installed via uv sync)
uv add huggingface_hub

# Login to HuggingFace (you'll need to accept GAIA's terms first)
huggingface-cli login

# Download the dataset
python -c "
from huggingface_hub import snapshot_download
data_dir = snapshot_download(
    repo_id='gaia-benchmark/GAIA',
    repo_type='dataset',
    local_dir='./data/gaia'
)
print(f'Dataset downloaded to: {data_dir}')
"
```

> **Note**: You must first accept the dataset terms at https://huggingface.co/datasets/gaia-benchmark/GAIA

### 4. Run Agent on Single Task

```bash
uv run python -m src.cli run --task-id <task_id>
```

### 5. Run Benchmark Evaluation

```bash
uv run python -m src.cli benchmark --level 1 --max-tasks 20
```

## Project Structure

```
src/
├── config.py         # Configuration management
├── agent.py          # Main autonomous agent
├── cli.py            # Command-line interface
├── tools/            # Agent tools
│   ├── e2b_executor.py
│   ├── rag.py
│   ├── web_search.py
│   └── file_handler.py
├── planning/         # Planning & state
│   ├── planner.py
│   └── state.py
└── benchmark/        # GAIA integration
    ├── gaia_loader.py
    ├── runner.py
    └── evaluator.py
```

## Architecture

The agent uses Anthropic's Agent SDK with custom tools exposed as in-process MCP servers:

- **E2B Executor**: Sandboxed Python execution
- **RAG System**: Vector-based knowledge retrieval
- **Web Search**: Generic HTTP-based information retrieval
- **File Handler**: Multi-format file parsing

## License

MIT
