<<<<<<< HEAD
# Agentex – Phase 1: Personal AI Assistant

A local-first personal AI assistant powered by **Ollama and Qwen3 4B**, built with Python and Streamlit. Phase 1 establishes the foundation for a privacy-focused AI assistant with persistent conversations, multi-turn interactions, and locally hosted language model inference.

## Overview

Agentex is an evolving personal AI assistant designed to run locally without depending on paid LLM APIs.

Phase 1 focuses on building a reliable conversational assistant that can interact with a locally hosted language model, maintain conversation history, and provide a simple chat interface.

The project is designed to evolve into a more capable AI agent with tool calling, task automation, contextual memory, and external integrations in subsequent phases.

## Phase 1 – Objectives

- Build a local conversational AI assistant.
- Integrate a locally hosted LLM using Ollama.
- Develop a Streamlit-based chat interface.
- Maintain persistent conversation history.
- Support multi-turn conversations with contextual continuity.
- Handle model responses and reasoning-output filtering.
- Establish a modular and testable project architecture.

## Features

### 1. Local LLM Integration
- Uses Ollama as the local LLM runtime.
- Integrates Qwen3 4B for natural language understanding and response generation.
- Runs inference locally without requiring external LLM API calls.

### 2. Interactive Chat Interface
- Streamlit-based conversational interface.
- Supports sending prompts and displaying assistant responses.
- Provides a straightforward interface for interacting with the local model.

### 3. Persistent Conversation History
- Uses SQLite for conversation storage.
- Maintains conversation records across application sessions.
- Supports retrieving previous conversations.

### 4. Multi-turn Conversations
- Maintains conversational context across multiple interactions.
- Supports follow-up questions based on previous messages.

### 5. Response Processing
- Handles model-generated responses.
- Filters unwanted reasoning-related output from user-facing responses.
- Includes error handling for model communication and response processing.

### 6. Testing
- Unit tests for core assistant functionality.
- Tests for conversation memory and multi-turn interactions.
- Ollama client and response-processing tests.
- Integration tests for live Ollama connectivity.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| LLM Runtime | Ollama |
| Language Model | Qwen3 4B |
| User Interface | Streamlit |
| Database | SQLite |
| Testing | Pytest |
| Environment Management | Conda |
| Version Control | Git and GitHub |

## Architecture

The application follows a modular architecture to separate the user interface, assistant logic, LLM communication, and conversation storage.

```text
                  User
                   |
                   v
          Streamlit Interface
                 app.py
                   |
                   v
            Assistant Logic
             agent/agent.py
                   |
                   v
             Ollama Client
           llm/ollama_client.py
                   |
                   v
              Ollama
              Qwen3 4B

                   |
                   v
           Response Processing
            llm/think_filter.py
                   |
                   v
             User Response

        Conversation Persistence
                   |
                   v
             SQLite Database
              database/
                   |
                   v
            Memory Management
                memory/
```

## Project Structure

```text
personal_ai_assistant/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   └── prompts.py
│
├── data/
│
├── database/
│   ├── __init__.py
│   ├── db.py
│   └── schema.sql
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
│   └── unit/
│
├── utils/
│   ├── __init__.py
│   └── logger.py
│
├── .gitignore
├── app.py
├── config.py
├── main.py
├── pytest.ini
├── requirements.txt
├── requirements.lock.txt
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.11
- Conda or Miniconda
- Ollama
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd personal_ai_assistant
```

### 2. Create and Activate Conda Environment

```bash
conda create -n personal_agent python=3.11 -y
conda activate personal_agent
```

### 3. Install Dependencies

For reproducible installation using the locked package versions:

```bash
pip install -r requirements.lock.txt
```

Alternatively, install the declared project dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Install Ollama from:

https://ollama.com/

Download the Qwen3 4B model:

```bash
ollama pull qwen3:4b
```

Start the Ollama service if it is not already running.

Verify the installed model:

```bash
ollama list
```

### 5. Run the Application

From the project root:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

## Running Tests

Run the unit tests:

```bash
pytest tests/unit -v
```

Run the complete test suite:

```bash
pytest -v
```

**Note:** Integration tests that communicate with Ollama may require the Ollama service to be running and the configured model to be available.

## Current Limitations

Phase 1 focuses on conversational functionality. The following capabilities are planned for future development rather than included as completed Phase 1 features:

- Autonomous tool selection and execution.
- Multi-step agent workflows.
- Task scheduling and reminders.
- Advanced long-term memory.
- External service integrations.
- Document analysis and summarization.

## Development Roadmap

| Phase | Planned Development | Status |
|---|---|---|
| Phase 1 | Local conversational assistant, Ollama integration, persistent conversations | Implemented |
| Phase 2 | LangGraph agent workflow and tool calling | Planned |
| Phase 3 | Task management and reminders | Planned |
| Phase 4 | Advanced short-term and long-term memory | Planned |
| Phase 5 | Weather, document assistance, and external integrations | Planned |
| Phase 6 | Performance optimization, testing, and improvements | Planned |

## Design Principles

- **Local-first:** Prioritize locally hosted models and services.
- **Privacy-focused:** Keep conversations and processing on the local machine wherever possible.
- **Modular architecture:** Separate application components for maintainability and extensibility.
- **Open-source ecosystem:** Prefer freely available tools and technologies.
- **Extensibility:** Build a foundation that can evolve into a capable personal AI agent.

## Future Vision

Agentex aims to evolve beyond a conventional chatbot into a personal AI agent capable of understanding requests, selecting appropriate tools, executing multi-step tasks, retaining useful context, and assisting with everyday activities.

Phase 1 establishes the conversational foundation. Future phases will progressively introduce agentic capabilities and automation.

---

**Project:** Agentex  
**Current Release:** Phase 1  
**Architecture:** Local-first AI Assistant  
**License:** To be determined
=======
# Personal AI Assistant (local, free, CPU-friendly)

A local AI assistant built on Ollama (Qwen3 4B), Streamlit and SQLite. No API keys, no paid services.
**Current status: Phase 2 – a LangGraph agent with a calculator tool, on top of Phase 1 chat with persistent
conversations.** Tasks, reminders, notes and weather arrive in Phases 3–5.

## Windows setup

1. Install [Ollama for Windows](https://ollama.com/download) and Miniconda/Anaconda.
2. Open **Anaconda Prompt** in this folder and run:

```
conda create -n personal_agent python=3.11 -y
conda activate personal_agent
pip install -r requirements.txt      # Phase 2 adds langgraph, so rerun this after upgrading
ollama pull qwen3:4b
streamlit run app.py
```

Optional checks:

```
python main.py --check     # is Ollama running and the model installed?
python main.py             # chat in the terminal
pytest                     # unit tests (no Ollama needed)
pytest -m integration      # live tests (Ollama must be running)
```

## Configuration (environment variables, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | Model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_THINK` | `false` | Qwen3 thinking mode (much slower on CPU) |
| `OLLAMA_NUM_CTX` | `4096` | Context window (higher = more RAM) |
| `OLLAMA_NO_THINK_SUFFIX` | `true` | Append Qwen3's `/no_think` switch to prompts while thinking is off (sent to the model only, never saved) |
| `OLLAMA_REASONING_HOLD_CHARS` | `20000` | Max characters held back while checking for reasoning (about a whole reply); `0` = stream word by word |
| `AGENT_ENABLED` | `true` | `false` = Phase 1 plain streaming chat (no tools, LangGraph not loaded) |
| `AGENT_MAX_ITERATIONS` | `5` | Max model calls per request (loop protection) |
| `AGENT_MAX_TOOL_CALLS_PER_TURN` | `4` | Extra tool calls in one model turn are skipped |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long the model stays loaded in RAM after a reply |
| `MAX_HISTORY_CHARS` | `6000` | Character budget for history sent to the model |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Request timeout |
| `MAX_HISTORY_MESSAGES` | `20` | Past messages sent to the model |
| `ASSISTANT_DB_PATH` | `data/assistant.db` | SQLite file (created automatically) |
| `LOG_LEVEL` | `INFO` | Logging level |

Set one in Anaconda Prompt: `set OLLAMA_MODEL=qwen3:1.7b` (before `streamlit run app.py`).

## How the modules fit together

| Module | Purpose |
|---|---|
| `config.py` | Central settings from environment variables |
| `app.py` | Streamlit UI only; no business logic |
| `main.py` | Terminal entry point and connection check |
| `agent/agent.py` | `Assistant.respond_stream()`: builds the prompt from history, runs the agent (or plain chat), reports tool progress, saves both turns |
| `agent/graph.py` | The LangGraph graph: `agent` -> `tools` -> `agent` ... with an iteration limit |
| `agent/state.py` | Agent state (messages + iteration counter) |
| `agent/tool_parsing.py` | Validated fallback for tool calls a model writes as text |
| `tools/registry.py` | Describes tools to the model; validates arguments; contains tool failures |
| `tools/calculator_tool.py` | Safe calculator (own parser, exact decimals, never `eval`) |
| `agent/prompts.py` | System prompt |
| `llm/ollama_client.py` | Ollama health check, model creation, streaming, error translation, `<think>` filtering |
| `memory/conversation_memory.py` | Conversations and messages in SQLite, always scoped by conversation id |
| `database/db.py`, `schema.sql` | Engine creation; `schema.sql` is run at startup, so no manual setup |
| `utils/logger.py` | Console and rotating file log (`data/assistant.log`) |

Flow: UI → `Assistant` → recent history from SQLite + system prompt → Ollama (streamed) → UI; after a
successful reply both messages are saved. If the model fails nothing is saved, and the UI shows how to fix it.
Streamlit session state only holds the *current conversation id*; all content lives in SQLite.

## Notes and limitations

- Only the calculator exists so far; tasks, reminders, notes and weather arrive in later phases.
- Qwen3 supports tool calling in Ollama, but a 4B model can be unreliable at it. Phase 2 will test this and
  add a validated fallback if needed, rather than trusting any JSON the model emits.
- Reminders will not fire while the app is closed; overdue ones will be listed on next start.

## Phase 2: how the agent works

```
user message
   |
   v
 [agent]  one Qwen3 turn with the calculator described to it
   |-- no tool needed ------------------------------> final answer -> shown + saved
   |-- tool call(s) --> [tools] validate args, run, capture result/error
   |                       |
   |                       +--> back to [agent] with the tool result
   +-- iteration limit reached --> [limit] safe message -> shown + saved
```

- The **model decides** whether a tool is needed (structured tool calling). There is no keyword routing, so
  "Who is Cristiano Ronaldo?" is answered directly and "Calculate 15% of 8500" goes to the calculator.
- The calculator accepts `+ - * / ^`, parentheses, decimals and percentages (`15%` = 15/100;
  `15% of 8500` and `8500 * 15%` both give 1275). Results are exact to 10 decimals. Division by zero,
  malformed input, huge exponents and anything that is not arithmetic are rejected with a clear message.
- A tool error is returned to the model as text so it can explain the problem or retry once. Failures
  inside a tool never crash the app and never show a stack trace.
- **Loop protection:** at most `AGENT_MAX_ITERATIONS` model calls per request, at most
  `AGENT_MAX_TOOL_CALLS_PER_TURN` tool calls per model turn, plus a LangGraph recursion limit as a backstop.
- **History:** only your message and the final answer are saved. Tool traces, prompts and reasoning are not
  stored, and the UI shows only short tool steps (for example "Using calculator: `8500 * 15%`").
- If a small model prints a tool call as text instead of a structured call, `agent/tool_parsing.py` recovers
  it **only** when it is valid JSON naming a registered tool; everything else is ignored and hidden.
- Each model turn is a single complete request, so the answer appears when it is ready (as in Phase 1's
  default hold mode). Token-by-token streaming of agent turns is not available in Phase 2; set
  `AGENT_ENABLED=false` for Phase 1 streaming chat.

Verified-by-you checklist after upgrading (see the test commands below): the table in your Phase 2 brief
(direct questions vs. calculator questions) is covered by `pytest -m integration`, which uses the real model.

### Dependency lock file

No lock file is shipped because versions cannot be resolved offline. After everything works, create one with
`pip freeze > requirements.lock.txt` and keep it with the project.

### Phase 2 troubleshooting

- **The assistant does arithmetic itself instead of using the calculator:** run `ollama --version` (use a
  recent release) and `pip install -U langchain-ollama langgraph`. Qwen3 4B can occasionally skip a tool; rephrase
  ("Use the calculator to compute...") or retry. `pytest -m integration` shows how often it happens.
- **Raw `<tool_call>` text or JSON appears in an answer:** please report it; the fallback parser should hide it.
- **"The agent failed unexpectedly":** details are in `data/assistant.log`. Set `AGENT_ENABLED=false` to keep
  chatting with Phase 1 behaviour while you investigate.
- **`ModuleNotFoundError: langgraph`:** run `pip install -r requirements.txt` inside the `personal_agent` env.
- **"I couldn't finish this request within the allowed number of steps":** the model kept requesting tools;
  rephrase the question or raise `AGENT_MAX_ITERATIONS`.

## Reasoning leakage (Qwen3 "thinking" text in answers)

Reasoning is **off by default** and is removed in three layers:

1. `reasoning=False` (Ollama `think:false`) and the `/no_think` switch added to the model-bound prompt.
2. Only the reply's `content` is ever read. A separate `reasoning_content` field is never shown or saved.
3. `llm/think_filter.py` removes `<think>...</think>` blocks and also reasoning that ends in `</think>` with
   **no opening tag** (some Qwen3 variants pre-fill `<think>` in their template).

Trade-off: the filter can only tell reasoning from an answer when `</think>` arrives, so by default it
holds the reply back until that tag shows up or generation ends (up to `OLLAMA_REASONING_HOLD_CHARS`,
default 20000, roughly a full reply). Reasoning of any realistic length is therefore removed, and the text after
`</think>` streams normally. If the model produces no tags at all, the whole reply appears when it is finished.
If you switch to a model that never reasons (see below), set `OLLAMA_REASONING_HOLD_CHARS=0` to get word-by-word
streaming back.

Prompt-level guards (`agent/prompts.py`): the system prompt asks for the final answer only, short replies,
pronoun resolution from earlier messages, and a one-line "may be out of date" note for current facts. Every user
turn sent to a Qwen3 model (current and replayed history) carries `/no_think`; stored history stays clean.

**Known limitation:** reasoning written as ordinary text with *no* `</think>` or `<think>` tag cannot be told
apart from an answer, so it cannot be removed. That means the model build is ignoring the thinking switch. The
reliable fixes are a non-thinking model build or `OLLAMA_THINK=true` (Ollama then returns reasoning in a separate
field that this app ignores; slower on CPU).

Still leaking? Run `python scripts/diagnose_ollama.py` and check:
- `langchain-ollama` is 0.3.4 or newer and Ollama is 0.9 or newer (older servers ignore `think`).
- `ollama show qwen3:4b`: if it is a thinking-only build, `think:false` may not be honoured. Check
  ollama.com/library/qwen3 for an instruct (non-thinking) tag and set `OLLAMA_MODEL` to it.
- The warning "Removed N characters of reasoning" in `data/assistant.log` shows the filter is catching leaks.

## Troubleshooting

- **Ollama not running / connection refused:** start the Ollama app or run `ollama serve`; check `OLLAMA_BASE_URL`.
- **Model not found:** `ollama pull qwen3:4b`, then click *Re-check connection*.
- **Dependency conflicts:** use a fresh env: `conda remove -n personal_agent --all -y`, recreate, reinstall.
- **Insufficient RAM:** close other apps, lower `OLLAMA_NUM_CTX` (e.g. 2048), or use `qwen3:1.7b`.
- **Slow CPU inference:** keep `OLLAMA_THINK=false`, lower `MAX_HISTORY_MESSAGES`, use a smaller model.
  The first reply after startup is slowest because the model is loading.
>>>>>>> master
