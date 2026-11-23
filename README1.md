# NexaMind – Phase 1: Personal AI Assistant

A local-first personal AI assistant powered by **Ollama and Qwen3 4B**, built with Python and Streamlit. Phase 1 establishes the foundation for a privacy-focused AI assistant with persistent conversations, multi-turn interactions, and locally hosted language model inference.

## Overview

NexaMind is an evolving personal AI assistant designed to run locally without depending on paid LLM APIs.

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

NexaMind aims to evolve beyond a conventional chatbot into a personal AI agent capable of understanding requests, selecting appropriate tools, executing multi-step tasks, retaining useful context, and assisting with everyday activities.

Phase 1 establishes the conversational foundation. Future phases will progressively introduce agentic capabilities and automation.

---

**Project:** NexaMind  
**Current Release:** Phase 1  
**Architecture:** Local-first AI Assistant  
**License:** To be determined
