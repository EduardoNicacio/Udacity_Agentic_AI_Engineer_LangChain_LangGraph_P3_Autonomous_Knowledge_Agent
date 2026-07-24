# UDA-Hub - Autonomous Knowledge Agent

LangGraph-powered multi-agent decision system for customer support automation, built for **CultPass** - a cultural experience subscription platform.

Tickets flow through a 4-agent pipeline: classification, knowledge-base resolution (RAG), and either automated resolution or structured escalation to human support - with long-term memory accumulating across sessions.

## Architecture

```txt
START → Supervisor (entry) → Classifier → Resolver
    → confidence >= 0.6 → Supervisor (exit) → END
    → confidence <  0.6 → Escalation → Supervisor (exit) → END
```

| Agent | Role |
| :--- | :--- |
| **SupervisorAgent** | Entry/exit node. Loads long-term memory at start, composes final response and writes memory on completion. |
| **ClassifierAgent** | Assigns category (billing, account, technical, subscription, content, onboarding), urgency, and routing label via LLM structured output. |
| **ResolverAgent** | Searches knowledge base via MCP tool, generates response, computes confidence score (0.0–1.0). Resolves if >= 0.6, escalates otherwise. |
| **EscalationAgent** | Generates structured escalation summary with reason, priority, customer context, and suggested actions for human handoff. |

## MCP Tools

All database access goes through FastMCP tool servers via `langchain-mcp-adapters`. No direct SQLAlchemy calls in agent files.

| Tool | Description |
| :--- | :--- |
| `account_lookup` | Queries CultPass external DB for user profile + subscription data; cross-references UDA-Hub for open tickets. |
| `refund_tool` | Validates ticket status, processes refund, logs to RefundAction table. |
| `kb_search` | Searches Knowledge table via keyword/tag/content matching with minimum score threshold (>= 4). |
| `read_memory` / `write_memory` | Reads/writes ConversationMemory records for cross-session context. |

## Knowledge Base

18 articles across 6 categories seeded from `cultpass_articles.jsonl`:

- **Billing & refunds** (3) - refund requests, billing cycles, double charges
- **Account management** (3) - login/password, email updates, account deletion
- **Technical troubleshooting** (3) - app crashes, QR codes, streaming
- **Subscription plans** (3) - cancel/pause, upgrade, downgrade
- **Content & features** (2) - ratings, reviews
- **Onboarding** (2) - getting started, app setup

## Getting Started

### Prerequisites

- Python 3.11+
- Jupyter (for running setup notebooks)

### Installation

```bash
cd solution
pip install -r requirements.txt
```

### Database Setup

```bash
# 1. Launch Jupyter
jupyter notebook

# 2. Run the setup notebooks in order:
#    01_external_db_setup.ipynb   → creates cultpass.db (users, subscriptions, experiences, reservations)
#    02_core_db_setup.ipynb       → creates udahub.db   (KB articles, accounts, users, memory)
```

### Run the Agent

```bash
python 03_agentic_app.py
```

Enter a ticket ID (or press Enter for default), then type a customer issue. The agent will classify it, search the knowledge base, and either resolve or escalate.

## Testing

```bash
python -m pytest tests/ -v
```

26 tests across 4 modules, all using mocks (no live LLM calls):

| Module | Tests | Coverage |
| :--- | :--- | :--- |
| `test_classifier.py` | 9 | 6 categories + escalation routing + empty input + LLM failure fallback |
| `test_resolver.py` | 6 | Article matching, escalation on no match, tool results, empty input, low confidence, memory context |
| `test_tools.py` | 6 | account_lookup (email/id/not found/no input) + refund_tool (invalid ticket/negative amount) |
| `test_workflow.py` | 5 | End-to-end: resolved ticket, escalated ticket, state shape, billing, agent trace order |

## Demo Test Cases

| Ticket | Path | Outcome |
| :--- | :--- | :--- |
| "I was charged twice for my subscription" | Classifier → Resolver → Resolve | Confidence 0.75, resolved |
| "I can't log in, forgot my password" | Classifier → Resolver → Resolve | Confidence 0.9, resolved |
| "How do I reserve a spot for an event?" | Classifier → Resolver → Resolve | Confidence 0.9, resolved |
| "My pet giraffe ate my phone" | Classifier → Resolver → Escalation | Confidence 0.0, escalated |

## Project Structure

```txt
solution/
├── agentic/
│   ├── agents/          # 4 agent modules (supervisor, classifier, resolver, escalation)
│   ├── tools/           # 4 MCP tool servers (account_lookup, refund, kb_search, memory)
│   ├── design/
│   │   └── architecture.md
│   └── workflow.py      # StateGraph built from scratch
├── data/
│   ├── core/            # udahub.db (seeded by notebook)
│   ├── external/        # cultpass.db + cultpass_articles.jsonl (seeded by notebook)
│   └── models/          # SQLAlchemy models for both databases
├── tests/               # 26 mocked tests
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.py
├── utils.py
├── requirements.txt
└── .env.example
```

## Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration with `StateGraph` (built from scratch, no prebuilt workflows)
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration and prompt management
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP tool server framework
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) - Bridges MCP tools into LangChain agent calls
- [SQLAlchemy](https://www.sqlalchemy.org/) - ORM for dual-database architecture
- [OpenAI](https://platform.openai.com/) - GPT-4o-mini for classification, resolution, and escalation

## License

This project is part of the [Udacity](https://udacity.com) Agentic AI Engineer with LangChain and LangGraph Nanodegree program.
