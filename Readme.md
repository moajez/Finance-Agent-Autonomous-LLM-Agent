# Finance Agent — Autonomous Multi-Agent Financial Research System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-000000?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangChain DeepAgents](https://img.shields.io/badge/Harness-DeepAgents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://docs.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash_/_3_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Qdrant Hybrid Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant_Hybrid-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Model Context Protocol](https://img.shields.io/badge/Tool_Standard-MCP_Yahoo_Finance-7B1FA2?style=for-the-badge&logo=protocol&logoColor=white)](https://modelcontextprotocol.io/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange?style=for-the-badge&logo=databricks&logoColor=white)](https://smith.langchain.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **An enterprise-grade autonomous multi-agent research system designed for institutional-level financial intelligence.** Combines hierarchical supervisor-worker architectures, sandboxed filesystem context isolation, multimodal Deep RAG over SEC filings (10-K, 10-Q), real-time market data through the Model Context Protocol (MCP), and test-time reflective reasoning.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [The Problem: Why Naive LLM & Basic RAG Fail in Finance](#-the-problem-why-naive-llm--basic-rag-fail-in-finance)
- [System Architecture](#-system-architecture)
- [Evolution of the Agent Architecture (Notebooks 09–12)](#-evolution-of-the-agent-architecture)
  - [Phase 1: Autonomous Single-Agent with Dynamic ToDo Planner (NB 09)](#phase-1-autonomous-single-agent-with-dynamic-todo-planner-notebook-09)
  - [Phase 2: Hierarchical Multi-Agent System Built from Scratch (NB 10)](#phase-2-hierarchical-multi-agent-system-built-from-scratch-notebook-10)
  - [Phase 3: Production DeepAgent with Context Isolation & Reflection (NB 11)](#phase-3-production-deepagent-with-context-isolation--reflection-notebook-11)
  - [Phase 4: Full-Stack Production Deployment, UI & Tracing (NB 12)](#phase-4-full-stack-production-deployment-ui--tracing-notebook-12)
- [Technical Decisions: "Why Tech X over Tech Y"](#-technical-decisions-why-tech-x-over-tech-y)
- [Dual-Engine Financial Information Layer](#-dual-engine-financial-information-layer)
  - [1. Multimodal Deep RAG (SEC 10-K & 10-Q Filings)](#1-multimodal-deep-rag-engine)
  - [2. Model Context Protocol (MCP) for Live Market Intelligence](#2-model-context-protocol-mcp-engine)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Production Deployment](#-production-deployment)
- [Engineering Highlights for Recruiters & Hiring Managers](#-engineering-highlights)

---

## 🎯 Executive Summary

Financial analysis requires absolute precision, temporal awareness, cross-document reconciliation, and up-to-the-minute market data. Standard conversational LLMs hallucinate numbers, fail at tabular arithmetic across hundreds of SEC pages, and suffer context collapse when dealing with massive corporate disclosures.

This repository implements a **production-ready Autonomous Multi-Agent Financial Research System** that autonomously decomposes high-level equity research directives (e.g., *"Analyze Meta's 2024 CapEx acceleration in AI infrastructure from 10-K filings, contrast against quarterly revenue growth, and evaluate current market valuation multiples and analyst consensus"*), coordinates specialized sub-agents, retrieves grounded multimodal evidence from SEC filings, fetches live market metrics via MCP, performs deliberate test-time verification, and compiles institutional-grade investment memoranda.

---

## ❌ The Problem: Why Naive LLM & Basic RAG Fail in Finance

| Challenge in Financial AI | Why Naive Implementations Fail | How This System Solves It |
| :--- | :--- | :--- |
| **Complex Multi-Step Goals** | Standard LLMs attempt to answer immediately in one pass, hallucinating missing data points or forgetting parts of multi-faceted inquiries. | **Hierarchical Planning & Dynamic ToDo Middleware**: Decomposes queries into structured thematic sub-questions, executing each in dedicated sub-tasks. |
| **Financial Tables & Footnotes** | Standard vector chunking cuts tables in half, losing row headers, units (\$ millions), and footnotes. | **Docling Multimodal Parsing**: Preserves native Markdown table semantics, document hierarchy, and cross-references page numbers and bounding coordinates. |
| **Keyword vs. Semantic Duality** | Vector similarity fails on specific financial identifiers (ticker `MSFT`, line item `"Operating Income"`). Keyword search misses conceptual meaning. | **Hybrid Retrieval (Dense + BM25) + Structured Filter Extraction + Cross-Encoder Reranking (`BAAI/bge-reranker-base`)**. |
| **Context Window Degradation** | Stuffing dozens of retrieved 10-K sections into one prompt causes attention decay, needle-in-a-haystack failures, and massive token costs. | **Filesystem-as-Context & Context Isolation**: Intermediate research is offloaded to a sandboxed virtual filesystem; sub-agents run in isolated contexts with clean message stacks. |
| **Stale vs. Historical Data Gap** | RAG vector stores only know historical data; web search lacks audited SEC accuracy. | **Dual-Engine Architecture**: Qdrant hybrid vector store for historical filings + Model Context Protocol (MCP) Yahoo Finance client for real-time order books, analyst consensus, and live news. |

---

## 🏛 System Architecture

The following diagram illustrates the complete hierarchical multi-agent workflow, information engines, and storage layers:

```mermaid
flowchart TD
    User([User / Investment Analyst]) -->|Complex Financial Inquiry| Orchestrator[Orchestrator / Supervisor Agent]

    subgraph Memory & Context Layer
        Checkpointer[(SQLite State Checkpointer\nSqliteSaver - Persistent Memory)]
        VFS[(Sandboxed Virtual Filesystem\nFilesystemBackend virtual_mode=True\nresearch_plan.md | researcher/*.md | report.md)]
    end

    Orchestrator <--> Checkpointer
    Orchestrator <--> VFS

    subgraph Planning & Decomposition
        Planner[Thematic Decomposition & Dynamic ToDo Planner]
        Orchestrator --> Planner
        Planner -->|Generates 3-5 Thematic Questions| ResearchPlan[research_plan.md]
    end

    subgraph Specialized Worker Sub-Agents
        Researcher[Financial Researcher Sub-Agent\nContext-Isolated Worker]
        Editor[Editor / Synthesis Sub-Agent\nReport Writer]
    end

    Orchestrator -->|Dispatches Sub-Task| Researcher
    Researcher <--> VFS
    Researcher -->|Reflective Pause| ThinkTool[think_tool: Strategic Reflection & Gap Analysis]

    subgraph Dual-Engine Information Layer
        subgraph Historical Filings Engine: Multimodal Deep RAG
            LLMFilter[LLM Structured Filter Extractor\nTicker / DocType / FY / FQ]
            Qdrant[(Qdrant Vector Database\nDense: gemini-embedding-001\nSparse: Qdrant/bm25)]
            Reranker[Cross-Encoder Reranker\nBAAI/bge-reranker-base]
            DoclingDocs[Audited SEC 10-K, 10-Q, 8-K Filings\nParsed via Docling + Multimodal Charts]
        end

        subgraph Real-Time Market Engine: Model Context Protocol
            MCPClient[MultiServerMCPClient - stdio]
            MCPServer[yahoo-finance-mcp-server\nPrices | News | Financials | Options | Analyst Ratings]
        end
    end

    Researcher -->|hybrid_search| LLMFilter
    LLMFilter --> Qdrant
    DoclingDocs --> Qdrant
    Qdrant --> Reranker
    Reranker -->|Grounded Evidence & Page Citations| Researcher

    Researcher -->|live_finance_researcher| MCPClient
    MCPClient <--> MCPServer
    MCPServer -->|Live Ticker Data & News| Researcher

    Researcher -->|Writes Thematic Findings| VFS
    Orchestrator -->|All Themes Complete| Editor
    Editor -->|Reads research_plan.md & All researcher/*.md| VFS
    Editor -->|Compiles Final Verified Analysis| Report[Final Investment Report: report.md]
    Report --> Orchestrator
    Orchestrator -->|Streamed Response & Formatted Report| User
```

---

## 🚀 Evolution of the Agent Architecture

This project traces an engineering progression from foundational RAG to an enterprise-grade multi-agent autonomous harness across **Notebooks 09 through 12**:

### Phase 1: Autonomous Single-Agent with Dynamic ToDo Planner (Notebook 09)
*Found in `09 Deep Finance Researcher with TODO Planner.ipynb`*

- **Autonomous Task Management via Middlewares**: Implements LangChain's `TodoListMiddleware` to dynamically generate, track, and update internal task lists during multi-step execution.
- **Context Compaction via `SummarizationMiddleware`**: Solves context bloat in long conversations by automatically triggering summarization when history exceeds 25 messages, keeping the 10 most recent messages active while compressing earlier interactions.
- **Durable Session Persistence**: Configured with `SqliteSaver` checkpointer allowing stateful conversation sessions tracked by `thread_id`.
- **Integrated Dual Retrieval**: Grants the agent simultaneous access to `hybrid_search` (historical SEC filings) and `live_finance_researcher` (real-time market data).

```python
# Notebook 09 Architecture Pattern
agent = create_agent(
    model=model,
    tools=[hybrid_search, live_finance_researcher],
    system_prompt=MULTIMODEL_AGENT_PROMPT,
    checkpointer=SqliteSaver(conn=sqlite3.connect("data/todo_financial_research_agent.db")),
    middleware=[
        SummarizationMiddleware(model=model, trigger=[("messages", 25)], keep=("messages", 10)),
        TodoListMiddleware()
    ]
)
```

---

### Phase 2: Hierarchical Multi-Agent System Built from Scratch (Notebook 10)
*Found in `10 Multi-Agent Deep AI Finance Researcher.ipynb`*

- **Supervisor-Worker Paradigm**: Breaks monolithic execution into an explicit three-role hierarchy:
  1. **Orchestrator Agent**: High-level planner. Ingests user prompts, extracts 3–5 core financial themes, and writes `research_plan.md`. Dispatches each theme as an independent execution cycle.
  2. **Researcher Agent**: Deep worker. Executes targeted RAG queries and live market checks for exactly one theme at a time, persisting notes to disk with automated retries (`max_retries=2`).
  3. **Editor Agent**: Synthesis specialist. Operates after research concludes. Reads all compiled theme files, audits citations, resolves discrepancies, and writes `report.md`.
- **Filesystem-as-Context**: Solves token explosion by creating a shared filesystem workspace (`agent_files/<user_id>/<thread_id>/`). Instead of passing huge retrieval payloads in the conversation history, agents write structured Markdown artifacts and pass lightweight file hashes and status notifications via LangGraph `Command(update=...)`.
- **Custom Agent State (`DeepAgentState`)**: Strong schema extending `AgentState` with multi-tenant `user_id` and session-scoped `thread_id` parameters.

```python
# Custom Schema & File System Integration (Notebook 10)
class DeepAgentState(AgentState):
    user_id: NotRequired[str]
    thread_id: NotRequired[str]

@tool
def write_research_plan(thematic_questions: list[str], state: Annotated[DeepAgentState, InjectedState], ...):
    # Generates persistent research plan on disk
    ...

@tool
def run_researcher(theme_id: int, thematic_question: str, state: Annotated[DeepAgentState, InjectedState], ...):
    # Invokes isolated researcher worker and saves hash-indexed findings
    ...

@tool
def run_editor(state: Annotated[DeepAgentState, InjectedState]) -> str:
    # Synthesizes all disk artifacts into report.md
    ...
```

---

### Phase 3: Production DeepAgent with Context Isolation & Reflection (Notebook 11)
*Found in `11 Deep Agent - Multi-Agent Deep Finance Researcher.ipynb`*

- **LangChain `deepagents` Framework**: Implements modern enterprise agent harnesses via `create_deep_agent`.
- **True Context Isolation**: The research worker runs as an independent sub-agent (`financial-research-agent`). It possesses its own execution memory stack, preventing prompt bleed, tool output clutter, and instruction drift between the supervisor and worker.
- **Sandboxed Virtual Filesystem (`FilesystemBackend`)**: Enforces secure file operations with `virtual_mode=True`, establishing a strict sandbox boundary that prevents path traversal vulnerabilities while maintaining persistent session state.
- **Test-Time Compute via `think_tool`**: Introduces a strategic reflection mechanism. After every retrieval operation, the agent is forced to call `think_tool` to evaluate evidence quality, identify missing variables (e.g., *"Found 2024 revenue, but missing 2023 comparative operating margin"*), and decide deliberately whether to iterate or terminate search.

```python
# Production DeepAgent with Sandbox & Subagents (Notebook 11)
backend = FilesystemBackend(root_dir=folder, virtual_mode=True)

research_sub_agent = {
    "name": "financial-research-agent",
    "description": "Delegate financial research to this sub-agent. Give it one specific research task at a time.",
    "tools": [hybrid_search, live_finance_researcher, think_tool],
    "system_prompt": DEEP_RESEARCHER_INSTRUCTIONS,
}

deep_agent = create_deep_agent(
    model=model,
    tools=[hybrid_search, live_finance_researcher, think_tool],
    subagents=[research_sub_agent],
    backend=backend,
    system_prompt=DEEP_ORCHESTRATOR_INSTRUCTIONS,
    checkpointer=SqliteSaver(conn=conn)
)
```

---

### Phase 4: Full-Stack Production Deployment, UI & Tracing (Notebook 12)
*Found in `12 Deploy Multi Agent Deep RAG in Production.ipynb`*

- **Production Service Architecture**:
  - **LangGraph API Server**: Runs the agent graph as a persistent background service via `langgraph dev` (Port 2024). Supports streaming tokens, human-in-the-loop interruption, and background thread execution.
  - **Full-Stack UI**: Integrated with `@langchain-ai/deep-agents-ui` (Next.js 14, React, Tailwind CSS, Yarn on Port 3000), offering real-time agent step visualization, sub-agent thread inspectors, artifact previews, and interactive tool logs.
  - **Containerized Vector Store**: Qdrant running in Docker Compose with mounted persistent volume storage (`6333:6333`).
  - **Observability & Tracing via LangSmith**: Complete tracing of every token, tool call latency, cross-agent message payload, and reranking score for monitoring and eval pipelines.

---

## 💡 Technical Decisions: "Why Tech X over Tech Y"

### 1. Why LangGraph & DeepAgents over AutoGen or CrewAI?
- **Deterministic State Machines**: Financial workflows require strict execution bounds (planning must precede research; research must complete before editorial synthesis). LangGraph provides fine-grained DAG control flow, typed state transitions, and first-class SQLite checkpointing.
- **True Context Isolation**: CrewAI and early multi-agent frameworks often share or concatenate chat histories across agents, quickly exhausting context windows. `deepagents` isolates sub-agent message stacks and connects them through clean tool boundaries.
- **Production Infrastructure**: LangGraph offers production-grade servers (`langgraph dev` / LangGraph Cloud) with WebSockets, background task workers, and battle-tested React UI harnesses.

### 2. Why Model Context Protocol (MCP) for Live Market Data?
- **Standardized Tool Protocol**: Rather than writing brittle, custom wrappers around Yahoo Finance or Bloomberg APIs, MCP standardizes tool discovery, schema transmission, and execution over `stdio` or HTTP/SSE transports.
- **Decoupled Execution**: Tools run in isolated sub-processes via `uvx yahoo-finance-mcp-server`, isolating dependencies from the agent runtime and allowing seamless swapping of market data backends without touching agent code.

### 3. Why Hybrid Search (Dense + BM25) + Cross-Encoder Reranking?
- **Dense-Only Failure Mode**: Dense embeddings (e.g., `gemini-embedding-001`) map concepts well, but struggle to differentiate between `$14.5B` and `$15.4B`, or between `10-K` and `10-Q`.
- **Hybrid Fusion (Dense + BM25)**: Combining dense vector cosine similarity with BM25 sparse lexical tokens in Qdrant ensures both thematic relevance and exact keyword hits.
- **LLM Structured Filter Extraction**: Before search execution, Gemini extracts structured metadata (`company_name`, `doc_type`, `fiscal_year`, `fiscal_quarter`) into a typed Pydantic schema (`ChunkMetadata`), translating natural language into rigid Qdrant payload filters.
- **Cross-Encoder Reranking (`BAAI/bge-reranker-base`)**: Vector similarity calculates independent scores; cross-encoders compute joint attention across `(query, document)` pairs, eliminating irrelevant paragraphs and guaranteeing top-k precision.

### 4. Why Filesystem-as-Context over Infinite Context Windows?
- Large context windows (1M+ tokens) are expensive, slow, and suffer from "lost-in-the-middle" recall degradation.
- Storing research notes, drafts, and sources in structured files (`research_plan.md`, `researcher/*.md`, `report.md`) allows agents to write, inspect line-by-line (`read_file` with offset/limit), modify, and summarize findings in a token-efficient, human-auditable manner.

### 5. Why Test-Time Compute via `think_tool`?
- Autonomous agents often suffer from "premature convergence"—stopping after the first superficial search result.
- The `think_tool` enforces a cognitive pause where the model audits its gathered facts against the user's acceptance criteria, catalogs information gaps, and plans precise follow-up queries.

---

## 🔍 Dual-Engine Financial Information Layer

```
                          ┌──────────────────────────────────────────────┐
                          │         User / Agent Query Engine            │
                          └──────────────────────┬───────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
      ┌──────────────────────────────┐                       ┌──────────────────────────────┐
      │     Historical SEC Engine    │                       │    Real-Time Market Engine   │
      │    (Multimodal Deep RAG)     │                       │     (Model Context Protocol) │
      └──────────────┬───────────────┘                       └──────────────┬───────────────┘
                     │                                                       │
        • Docling Multi-Format Parser                          • MultiServerMCPClient (stdio)
        • Gemini Embedding-001 (Dense)                         • uvx yahoo-finance-mcp-server
        • FastEmbed BM25 (Sparse)                              • Live Quotes & Bid/Ask Spreads
        • Qdrant Hybrid Collection                             • Options Chains & Expirations
        • Pydantic Metadata Filtering                          • Financial Statement Breakdowns
        • BGE-Reranker Cross-Encoder                           • Latest Curated Financial News
```

### 1. Multimodal Deep RAG Engine
- **Source Documents**: High-resolution 10-K, 10-Q, and 8-K filings for major tech companies (Apple, Microsoft, Google, Meta, Amazon, Tesla, Nvidia) stored in `data/rag-data/`.
- **Docling Extraction**: Deconstructs raw PDFs into structured markdown, extracting complex multi-column balance sheets, income statements, and footnotes without OCR errors.
- **Multimodal Chart Intelligence**: Visual charts and infographics parsed with Gemini Multimodal to generate detailed analytical descriptions stored alongside table markdown.
- **Qdrant Vector Engine**: Configured in hybrid retrieval mode combining dense and sparse vectors with metadata payload indexes.

### 2. Model Context Protocol (MCP) Engine
- Powered by `langchain-mcp-adapters` connecting to `yahoo-finance-mcp-server`.
- Exposes 9 high-utility financial tools to the agent:
  - `get_historical_stock_prices`: Customizable period (`1mo`, `1y`) and interval (`1d`, `1wk`).
  - `get_stock_info`: Comprehensive market cap, P/E ratio, enterprise value, beta, forward EPS.
  - `get_yahoo_finance_news`: Real-time news sentiment and market catalysts.
  - `get_financial_statement`: Direct access to balance sheets, income statements, and cash flows.
  - `get_holder_info`: Institutional holders, mutual fund holdings, and insider transactions.
  - `get_option_chain` & `get_option_expiration_dates`: Options Greeks, strike volumes, put/call ratios.
  - `get_recommendations`: Wall Street analyst ratings, target prices, and upgrades/downgrades.

---

## 📂 Project Directory Structure

```plaintext
Finance-Agent-Autonomous-LLM-Agent/
├── 02 Gemini and Langsmith Setup.ipynb        # Environment configuration and LangSmith tracing setup
├── 03 Gemini3 with Langchain Bootcamp.ipynb      # Tool calling, structured outputs, and prompt patterns
├── 04 Langchain Agents Bootcamp.ipynb           # ReAct agent baselines and checkpointer basics
├── 05 MCP for Finance - Stock Researcher.ipynb  # Model Context Protocol (MCP) integration with Yahoo Finance
├── 06-01 Data Extraction with Docling.ipynb     # SEC filing parsing into markdown, tables, and images
├── 06-02 Image Descriptions.ipynb               # Multimodal chart description generation with Gemini
├── 06-03 Data Ingestion for Deep RAG.ipynb      # Hybrid vector store indexing in Qdrant (Dense + BM25)
├── 07 Retrieval for Deep RAG.ipynb              # Retrieval benchmarking, metadata filtering & BGE reranking
├── 08 Multimodal Research Agent.ipynb           # Image-augmented financial analysis agent
├── 09 Deep Finance Researcher with TODO.ipynb   # [PHASE 1] Single agent + ToDo Planner & Summarization
├── 10 Multi-Agent Deep AI Finance Researcher.ipynb # [PHASE 2] Hierarchical multi-agent system from scratch
├── 11 Deep Agent - Multi-Agent Deep Researcher.ipynb # [PHASE 3] Production DeepAgent with Context Isolation
├── 12 Deploy Multi Agent Deep RAG in Prod.ipynb # [PHASE 4] Full-stack LangGraph server & Next.js UI deployment
├── docker-compose.yml                           # Qdrant vector database container configuration
├── requirements.txt                             # Production Python dependencies
├── rag-flow.md                                  # Architectural documentation of the RAG data pipeline
├── Demo.gif                                     # UI interaction walkthrough and terminal demonstration
├── scripts/
│   ├── __init__.py
│   ├── agent_utils.py                           # Real-time streaming response formatters
│   ├── base_tools.py                            # Utility search and calculation tools
│   ├── deep_prompts.py                          # System prompts for DeepAgent orchestrator & researchers
│   ├── file_tools.py                            # Sandboxed disk filesystem tools (ls, read_file, write_file)
│   ├── prompts.py                             # Multi-agent system prompts (Orchestrator, Researcher, Editor)
│   ├── rag_tools.py                             # Hybrid search, structured filter extractor, and think_tool
│   ├── schema.py                                # Pydantic metadata schemas (DocType, ChunkMetadata)
│   └── yahoo_mcp.py                             # MultiServerMCPClient Yahoo Finance wrapper
└── data/
    ├── images/                                  # Financial infographics and visualizations
    ├── rag-data/                                # Parsed SEC 10-K, 10-Q markdown, tables, and images
    └── todo_financial_research_agent.db         # SQLite persistent state checkpoint database
```

---

## 🛠 Getting Started & Installation

### Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (for Qdrant vector database)
- **uv** (recommended) or `pip`
- **Node.js 20+ & Yarn** (optional, for frontend UI)

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/moajez/Finance-Agent-Autonomous-LLM-Agent.git
cd Finance-Agent-Autonomous-LLM-Agent

# Using uv (recommended)
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_google_gemini_api_key

# LangSmith Observability (Tracing & Monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Finance-Agent-Autonomous-Research

# Qdrant Vector DB Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=financial_docs

# Agent Workspace Root
AGENT_FILE_BASE_DIR=agent_files
```

### 3. Launch Qdrant Vector Store

```bash
docker compose up -d
```
*Verify Qdrant is running at `http://localhost:6333/dashboard`.*

### 4. Run the Multi-Agent Research System

You can run the interactive notebooks directly:
```bash
jupyter lab
```
Open and execute:
- `09 Deep Finance Researcher with TODO Planner.ipynb` for single-agent ToDo execution.
- `10 Multi-Agent Deep AI Finance Researcher.ipynb` for the hierarchical multi-agent supervisor system.
- `11 Deep Agent - Multi-Agent Deep Finance Researcher.ipynb` for the sandboxed DeepAgent system.

Or invoke the agent directly via Python:
```python
from scripts.rag_tools import hybrid_search, live_finance_researcher, think_tool
from scripts.deep_prompts import DEEP_RESEARCHER_INSTRUCTIONS, DEEP_ORCHESTRATOR_INSTRUCTIONS
from scripts.agent_utils import stream_agent_response
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Initialize model & persistence
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
conn = sqlite3.connect("data/agent_memory.db", check_same_thread=False)

# Configure sandboxed filesystem backend
backend = FilesystemBackend(root_dir="agent_files/session_1", virtual_mode=True)

# Build subagent & orchestrator
research_sub_agent = {
    "name": "financial-research-agent",
    "description": "Specialized sub-agent for SEC filings and live market research.",
    "tools": [hybrid_search, live_finance_researcher, think_tool],
    "system_prompt": DEEP_RESEARCHER_INSTRUCTIONS,
}

agent = create_deep_agent(
    model=model,
    tools=[hybrid_search, live_finance_researcher, think_tool],
    subagents=[research_sub_agent],
    backend=backend,
    system_prompt=DEEP_ORCHESTRATOR_INSTRUCTIONS,
    checkpointer=SqliteSaver(conn=conn)
)

# Stream agent execution
stream_agent_response(
    agent,
    query="Analyze Apple's Services revenue growth over the past 2 years from 10-K filings and compare with its current market valuation.",
    thread_id="thread_001"
)
```

---

## 🌐 Production Deployment

For full-stack deployment with streaming UI:

### 1. Launch the LangGraph API Server
```bash
langgraph dev --port 2024
```
*Exposes the LangGraph service with streaming token endpoints, checkpoint management, and health checks at `http://localhost:2024`.*

### 2. Launch the Full-Stack Agent UI
```bash
git clone https://github.com/langchain-ai/deep-agents-ui.git
cd deep-agents-ui
yarn install
yarn dev
```
*Open `http://localhost:3000` to interact with the visual multi-agent console, inspect generated research files, view tool executions in real-time, and download synthesized reports.*

---

## 👨‍💻 Engineering Highlights

### Why This Project Proves Senior-Level LLM Engineering Competence:

1. **Complex System Orchestration**: Did not settle for a basic single-prompt script or toy ReAct loop. Architected a multi-tier hierarchical system featuring plan-and-execute decomposition, supervisory delegation, and dedicated synthesis sub-agents.
2. **Context Engineering & Scalability**: Recognized that long multi-step research exhausts context windows and degrades attention. Engineered a filesystem-backed context mechanism where sub-agents offload intermediate notes to disk and read specific slices via line offsets.
3. **Advanced Information Retrieval**: Built a production-grade RAG stack: document layout parsing with Docling, hybrid dense/sparse vector representation in Qdrant, dynamic LLM metadata filter extraction, and cross-encoder reranking (`bge-reranker-base`).
4. **Tool Protocol Integration (MCP)**: Utilized the modern Anthropic/open-standard Model Context Protocol to interface with external market data servers, decoupling tool logic from the core orchestrator.
5. **Stateful Reliability & Guardrails**: Integrated durable SQLite checkpointing for thread resumption, exponential backoff retries on worker failures, and test-time cognitive reflection (`think_tool`).
6. **Observability & Deployment**: Baked in end-to-end telemetry via LangSmith and containerized data services with Docker Compose, targeting real production workloads rather than prototype notebooks.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with precision for enterprise AI & financial intelligence. Star ⭐ this repository if you find it valuable!</sub>
</div>
