# Agentex – Local-First Personal AI Assistant

**Agentex** is a local-first personal AI assistant designed to run on your own machine using open-source technologies. It uses Ollama and Qwen3 4B for local LLM inference, Streamlit for the user interface, SQLite for persistent conversation storage, and LangGraph for agent orchestration.

The project is being developed incrementally, evolving from a conversational assistant into an intelligent agent capable of using tools and performing tasks.

**Current Status: Phase 2 – LangGraph Agent and Calculator Tool Integration**

---

## 1. Project Objectives

- Build a personal AI assistant that runs locally.
- Avoid dependency on paid LLM APIs.
- Maintain persistent conversation history.
- Introduce an agent capable of deciding when to use tools.
- Support multi-step agent execution.
- Build a modular architecture that can accommodate additional tools and capabilities.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| LLM Runtime | Ollama |
| Language Model | Qwen3 4B |
| User Interface | Streamlit |
| Agent Framework | LangGraph |
| LLM Integration | LangChain Ollama |
| Database | SQLite |
| Testing | Pytest |
| Environment Management | Conda |

## 3. Implemented Features

### Phase 1 – Local Conversational Assistant

- Streamlit-based chat interface.
- Local LLM integration using Ollama.
- Persistent conversation storage using SQLite.
- Create, switch, and manage conversations.
- Conversation history retrieval.
- Conversation isolation using conversation IDs.
- Ollama connection and model availability checks.
- Error handling for connection and generation failures.
- Qwen3 reasoning-text filtering.

### Phase 2 – Agent Workflow and Tool Calling

- LangGraph-based agent workflow.
- LLM-directed tool selection.
- Calculator tool integration.
- Structured tool-calling support.
- Fallback parsing for tool calls returned as text.
- Tool argument validation.
- Safe mathematical expression evaluation without Python `eval()`.
- Multi-step agent execution.
- Tool execution results returned to the agent.
- Tool error handling and recovery.
- Iteration limits and recursion protection.
- Tool execution progress displayed in the interface.
- Conversation persistence for user messages and final assistant responses.

### Calculator Capabilities

The calculator supports:

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)
- Exponentiation (`^`)
- Parentheses
- Decimal values
- Percentage calculations

Examples:

```text
15% of 8500
8500 * 15%
(250 + 150) * 2
1000 / 4
2 ^ 10
```

The calculator validates expressions and rejects invalid mathematical operations, including division by zero and excessively large exponents.

## 4. Project Architecture

```text
Agentex/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── graph.py
│   ├── prompts.py
│   ├── state.py
│   └── tool_parsing.py
│
├── database/
│   ├── __init__.py
│   ├── db.py
│   └── schema.sql
│
├── data/
│   └── .gitkeep
│
├── llm/
│   ├── __init__.py
│   ├── ollama_client.py
│   └── think_filter.py
│
├── memory/
│   ├── __init__.py
│   └── conversation_memory.py
│
├── scripts/
│   ├── __init__.py
│   └── diagnose_ollama.py
│
├── tests/
│   ├── integration/
│   │   ├── test_agent_live.py
│   │   └── test_ollama_live.py
│   │
│   └── unit/
│       ├── test_agent_graph.py
│       ├── test_assistant.py
│       ├── test_calculator.py
│       ├── test_conversation_memory.py
│       ├── test_multi_turn.py
│       ├── test_ollama_client.py
│       ├── test_prompts.py
│       ├── test_registry.py
│       ├── test_think_filter.py
│       └── test_tool_parsing.py
│
├── tools/
│   ├── __init__.py
│   ├── calculator_tool.py
│   └── registry.py
│
├── utils/
│   ├── __init__.py
│   └── logger.py
│
├── app.py
├── config.py
├── main.py
├── pytest.ini
├── requirements.txt
├── requirements.lock.txt
├── .gitignore
└── README.md
```

## 5. Agent Workflow

The Phase 2 workflow uses LangGraph to coordinate the LLM and available tools.

```text
                  User Message
                       |
                       v
                 Agent Node
                       |
              LLM decides whether
                 a tool is needed
                       |
              +--------+--------+
              |                 |
          No Tool            Tool Call
              |                 |
              v                 v
         Final Answer      Tool Validation
              |                 |
              |                 v
              |            Tool Execution
              |                 |
              |                 v
              |          Tool Result Returned
              |                 |
              |                 v
              |             Agent Node
              |                 |
              +--------+--------+
                       |
                       v
                 Final Response
                       |
                       v
                SQLite Storage
```

The model decides whether a tool is required based on the user's request. The workflow does not depend on fixed keyword-based routing.

For example:

- General knowledge questions can be answered directly.
- Mathematical questions can be delegated to the calculator.
- Tool results are returned to the agent so it can formulate the final response.

## 6. Installation and Setup

### Prerequisites

- Windows
- Python 3.11
- Miniconda or Anaconda
- Ollama

### Step 1 – Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Agentex.git
cd Agentex
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2 – Create and activate the environment

```bash
conda create -n personal_agent python=3.11 -y
conda activate personal_agent
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Download the local model

```bash
ollama pull qwen3:4b
```

### Step 5 – Start the application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal.

## 7. Configuration

Agentex supports environment variables for configuring the LLM, agent workflow, and application.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_THINK` | `false` | Enables Qwen3 thinking mode |
| `OLLAMA_NUM_CTX` | `4096` | Context window |
| `AGENT_ENABLED` | `true` | Enables agent workflow |
| `AGENT_MAX_ITERATIONS` | `5` | Maximum model calls per request |
| `AGENT_MAX_TOOL_CALLS_PER_TURN` | `4` | Maximum tool calls per model turn |
| `OLLAMA_KEEP_ALIVE` | `30m` | Model keep-alive duration |
| `MAX_HISTORY_CHARS` | `6000` | Maximum conversation-history characters |
| `MAX_HISTORY_MESSAGES` | `20` | Maximum historical messages |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Model request timeout |
| `ASSISTANT_DB_PATH` | `data/assistant.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Logging level |

Configuration values are optional. The defaults can be overridden through environment variables.

## 8. Running Tests

Agentex uses Pytest for unit and integration testing.

### Run unit tests

```bash
pytest -m "not integration" -v
```

Unit tests do not require a running Ollama server.

### Run integration tests

Make sure Ollama is running and the configured model is available.

```bash
pytest -m integration -v
```

### Run all tests

```bash
pytest -v
```

## 9. Troubleshooting

**Ollama connection refused**

- Ensure the Ollama application is running.
- Verify the configured server URL.
- Run `ollama list` to check available models.

**Model not found**

```bash
ollama pull qwen3:4b
```

**LangGraph module not found**

```bash
pip install -r requirements.txt
```

**Agent execution failure**

- Check `data/assistant.log`.
- Verify Ollama availability.
- Set `AGENT_ENABLED=false` to temporarily use Phase 1 chat functionality.

**Insufficient memory or slow inference**

- Reduce `OLLAMA_NUM_CTX`.
- Consider using a smaller model such as `qwen3:1.7b`.
- Keep Qwen3 thinking mode disabled for faster CPU inference.

## 10. Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Local chatbot, Ollama integration, Streamlit UI, SQLite conversation persistence | Completed |
| Phase 2 | LangGraph agent, calculator tool, tool calling, execution controls | Implementation complete; validation ongoing |
| Phase 3 | Tasks, reminders, and note management | Planned |
| Phase 4 | Short-term and long-term memory improvements | Planned |
| Phase 5 | Additional tools such as weather and document summarization | Planned |
| Phase 6 | Testing, optimization, reliability, and performance improvements | Planned |

## 11. Design Principles

- Local-first execution.
- No mandatory paid API dependencies.
- Modular and maintainable code.
- Validated tool execution.
- Conversation isolation and persistence.
- Controlled agent execution.
- Test-driven development.
- Extensible architecture for future capabilities.

---

**Agentex is an evolving personal AI assistant project. The goal is to progressively develop it from a local conversational interface into a capable, tool-using personal agent.**