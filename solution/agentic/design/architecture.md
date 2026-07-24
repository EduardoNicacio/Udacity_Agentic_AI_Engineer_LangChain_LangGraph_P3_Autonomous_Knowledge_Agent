# UDA-Hub Architecture Design

## System Overview

UDA-Hub is a LangGraph-powered multi-agent decision system that processes customer support tickets through a pipeline of specialized agents. Each ticket flows through classification, support tool invocation, resolution attempt (with RAG), and either resolution or escalation - with long-term memory accumulating across sessions and scoped to individual customers.

The system is designed for CultPass, a cultural experience subscription platform, but is architected to be account-agnostic via the UDA-Hub multi-tenant data model.

## Agent Graph

```mermaid
graph TD
    START((START)) --> SUPERVISOR[supervisor_node<br/>Entry point + state init]
    SUPERVISOR --> CLASSIFIER[classifier_agent<br/>Category + urgency + routing]
    CLASSIFIER --> ROUTE_LABEL{route_by_label<br/>routing_label?}
    ROUTE_LABEL -->|resolver| SUPPORT_TOOLS[support_tools_node<br/>account_lookup + refund]
    ROUTE_LABEL -->|escalation| ESCALATION[escalation_agent<br/>Summarize for human handoff]
    SUPPORT_TOOLS --> RESOLVER[resolver_agent<br/>RAG lookup + confidence scoring]
    RESOLVER --> ROUTE_CONF{route_after_resolver<br/>confidence >= 0.6?}
    ROUTE_CONF -->|resolve| SUPERVISOR_FINAL[supervisor_final_node<br/>Compose response + write memory]
    ROUTE_CONF -->|escalate| ESCALATION
    ESCALATION --> SUPERVISOR_FINAL
    SUPERVISOR_FINAL --> END((END))

    style SUPERVISOR fill:#4a90d9,color:#fff
    style CLASSIFIER fill:#f5a623,color:#fff
    style SUPPORT_TOOLS fill:#50e3c2,color:#fff
    style RESOLVER fill:#7ed321,color:#fff
    style ESCALATION fill:#d0021b,color:#fff
    style SUPERVISOR_FINAL fill:#4a90d9,color:#fff
    style ROUTE_LABEL fill:#9b59b6,color:#fff
    style ROUTE_CONF fill:#9b59b6,color:#fff
```

### Node Descriptions

1. **supervisor_node** - Entry point. Receives raw ticket input, initializes state, loads customer-scoped long-term memory from `ConversationMemory` table via keyword/category matching.
2. **classifier_agent** - Reads ticket text + metadata. Uses LLM structured output to assign category, urgency, and routing label. Logs classification decision.
3. **route_by_label** - Conditional edge after classifier. If routing_label is "escalation", skip support_tools and resolver, go directly to escalation. If "resolver", proceed to support_tools.
4. **support_tools_node** - Invokes `account_lookup` (for billing/account tickets) and `process_refund` (for refund requests) MCP tools. Records tool results in state.
5. **resolver_agent** - Performs RAG against the Knowledge base (keyword/tag search with stemming and phrase matching). Selects best matching article, generates response, computes confidence score (0.0-1.0).
6. **route_after_resolver** - Conditional edge. If confidence >= 0.6 -> resolve. If < 0.6 -> escalate.
7. **escalation_agent** - Generates structured escalation summary for human handoff when routing_label is "escalation" or resolver confidence is low.
8. **supervisor_final_node** - Composes the final assistant response from resolution or escalation data. Writes customer-scoped long-term memory record. Appends to agent trace.

## Agent Specifications

### SupervisorAgent

| Property | Value |
|---|---|
| **File** | `solution/agentic/agents/supervisor_agent.py` |
| **Node functions** | `supervisor_node` (entry), `supervisor_final_node` (exit) |
| **Role** | Entry point that initializes state and loads customer-scoped long-term memory. Final node that composes the response and persists memory. |
| **Inputs** | Raw ticket text, user email, customer_id, thread_id |
| **Outputs** | Populated `memory_context` (entry), final `messages` + `ConversationMemory` write (exit) |
| **Tools** | `read_memory`, `write_memory` (MCP, loaded via `langchain-mcp-adapters`) |
| **LLM** | Used for composing final response only |

### ClassifierAgent

| Property | Value |
|---|---|
| **File** | `solution/agentic/agents/classifier_agent.py` |
| **Node function** | `classifier_node` |
| **Role** | Analyze ticket text + metadata to assign category, urgency, and routing label |
| **Inputs** | `ticket_text`, `user_email`, `memory_context` |
| **Outputs** | Updates `classification` in state: `{category, urgency, routing_label}` |
| **Tools** | None |
| **LLM** | Primary - uses structured output parsing |

**Classification categories:**

- `billing` - charges, refunds, payment issues
- `account` - login, password, profile, email changes
- `technical` - app crashes, QR code issues, streaming problems
- `subscription` - plan changes, upgrades, downgrades
- `content` - experience ratings, reviews, feature questions
- `onboarding` - getting started, app setup, first use

**Urgency levels:**

- `high` - financial loss, account lockout, data breach keywords
- `medium` - subscription changes, technical issues with workaround
- `low` - how-to questions, feature inquiries, general info

**Routing labels:**

- `resolver` - ticket can likely be answered from knowledge base
- `escalation` - requires human intervention (financial disputes, security, account deletion)

### SupportToolsAgent

| Property | Value |
|---|---|
| **File** | `solution/agentic/agents/support_tools_agent.py` |
| **Node function** | `support_tools_node` |
| **Role** | Invoke account_lookup and refund MCP tools based on ticket classification and content |
| **Inputs** | `ticket_text`, `classification`, `user_email`, `tool_results` |
| **Outputs** | Appends to `tool_results` in state |
| **Tools** | `account_lookup`, `process_refund` (MCP, loaded via `langchain-mcp-adapters`) |
| **LLM** | None |

**Tool invocation rules:**

- For billing tickets with user_email: invoke `account_lookup`
- For tickets mentioning "refund": invoke `process_refund`

### ResolverAgent

| Property | Value |
|---|---|
| **File** | `solution/agentic/agents/resolver_agent.py` |
| **Node function** | `resolver_node` |
| **Role** | Perform RAG against knowledge base, compute confidence, attempt resolution |
| **Inputs** | `ticket_text`, `classification`, `memory_context`, `tool_results` |
| **Outputs** | Updates `resolution` in state: `{status, article_id, article_title, response, confidence}` |
| **Tools** | `kb_search` (MCP, loaded via `langchain-mcp-adapters`) |
| **LLM** | Used for RAG response generation + confidence scoring |

### EscalationAgent

| Property | Value |
|---|---|
| **File** | `solution/agentic/agents/escalation_agent.py` |
| **Node function** | `escalation_node` |
| **Role** | Generate structured escalation summary for human handoff |
| **Inputs** | `ticket_text`, `classification`, `resolution` (partial), `memory_context` |
| **Outputs** | Updates `escalation` in state: `{reason, priority, customer_context, suggested_actions}` |
| **Tools** | None |
| **LLM** | Used for generating escalation summary |

## Routing Decision Table

### After Classifier (route_by_label)

| Routing Label | Next Node | Condition |
|---|---|---|
| `resolver` | support_tools_node | Ticket can be handled by KB + tools |
| `escalation` | escalation_node | Requires human intervention |

### After Resolver (route_after_resolver)

| Confidence | Action | Meaning |
|---|---|---|
| >= 0.6 | resolve -> supervisor_final | Strong enough KB match |
| < 0.6 | escalate -> escalation | Insufficient KB match |

### Combined Routing Examples

| Ticket Characteristic | Category | Urgency | Classifier Route | Confidence | Final Outcome |
|---|---|---|---|---|---|
| "How do I reserve an event?" | onboarding | low | resolver | 0.85 | Resolved |
| "Can't log in, forgot password" | account | medium | resolver | 0.75 | Resolved |
| "App crashes when reserving" | technical | medium | resolver | 0.80 | Resolved |
| "I want to upgrade to premium" | subscription | low | resolver | 0.90 | Resolved |
| "How do I leave a review?" | content | low | resolver | 0.70 | Resolved |
| "I was charged twice" | billing | high | resolver | 0.40 | Escalated (low confidence) |
| "I want a full refund and to cancel" | billing | high | escalation | - | Escalated (classifier) |
| "My account was hacked" | account | high | escalation | - | Escalated (classifier) |

## Confidence Scoring Specification

The confidence score is assigned by the LLM in the ResolverAgent.

**Threshold routing:**

| Score | Action | Meaning |
|---|---|---|
| 0.8-1.0 | Resolve | Strong KB match, high certainty |
| 0.6-0.79 | Resolve | Good match, adequate information |
| 0.3-0.59 | Escalate | Partial match, uncertain resolution |
| 0.0-0.29 | Escalate | No relevant article found |

## Memory Strategy

### Short-Term Memory (Session)

| Property | Value |
|---|---|
| **Storage** | LangGraph `MemorySaver` checkpointer |
| **Scope** | Thread ID (ticket_id passed as thread_id) |
| **Contents** | Message history (append-only via `operator.add`), intermediate agent outputs |
| **Access** | Automatic via state in workflow nodes |
| **Lifetime** | Duration of one session |
| **Multi-turn** | Each turn uses a unique thread_id; state is preserved via checkpointer |

### Long-Term Memory (Cross-Session)

| Property | Value |
|---|---|
| **Storage** | `ConversationMemory` table in UDA-Hub SQLite DB |
| **Schema** | memory_id, account_id, **customer_id**, ticket_id, summary, embedding, category, resolution_type, created_at |
| **Write trigger** | After successful resolution or escalation (in `supervisor_final_node`) |
| **Read trigger** | At session start (in `supervisor_node`) |
| **Retrieval method** | SQL LIKE queries on category + keyword overlap in summary, scoped by customer_id |
| **Customer scoping** | All reads/writes filtered by customer_id (= user_email) |
| **Limit** | Top 5 most relevant past interactions for the customer |

## Logging Strategy

### Log Format

Every agent step emits a single JSON line to stdout.

### What Gets Logged at Each Node

| Node | Actions Logged |
|---|---|
| `supervisor_node` | `ticket_received` (ticket_id, memory_count) |
| `classifier_node` | `classify_ticket` (category, urgency, routing_label) |
| `support_tools_node` | `support_tools_invoked` (invoked tools, category, tool_results_count) |
| `resolver_node` | `kb_search_resolved` (article_id, article_title, confidence, matches_found) |
| `escalation_node` | `escalation_created` (priority, reason) |
| `supervisor_final_node` | `session_complete` (resolution_type, category) |
| `router` (classifier) | `classifier_routing` (routing_label) |
| `router` (confidence) | `confidence_routing` (confidence, threshold, decision) |

## Tool Specifications

### account_lookup

| Property | Value |
|---|---|
| **File** | `solution/agentic/tools/account_lookup_tool.py` |
| **MCP Server** | `FastMCP("account_lookup_tool")` |
| **Input** | `email` OR `user_id` |
| **Output** | User + subscription info + open ticket count |
| **DBs accessed** | `cultpass.db` + `udahub.db` |
| **Invoked by** | `support_tools_node` for billing/account tickets |

### process_refund

| Property | Value |
|---|---|
| **File** | `solution/agentic/tools/refund_tool.py` |
| **MCP Server** | `FastMCP("refund_tool")` |
| **Input** | `ticket_id`, `amount`, `reason` |
| **Output** | Refund approval status + details |
| **DBs accessed** | `udahub.db` |
| **Invoked by** | `support_tools_node` for refund-related tickets |

### kb_search

| Property | Value |
|---|---|
| **File** | `solution/agentic/tools/kb_search_tool.py` |
| **MCP Server** | `FastMCP("kb_search")` |
| **Input** | `ticket_text`, `category`, `account_id` |
| **Output** | List of matching Knowledge articles with scores (min score 6) |
| **DBs accessed** | `udahub.db` |
| **Used by** | `resolver_agent` via `langchain-mcp-adapters` |
| **Ranking** | Stemming, phrase matching, title/tag/content weighted scoring, category alignment |

### read_memory / write_memory

| Property | Value |
|---|---|
| **File** | `solution/agentic/tools/memory_tool.py` |
| **MCP Server** | `FastMCP("memory_tool")` |
| **Input (read)** | `category`, `ticket_text`, `limit`, `account_id`, **`customer_id`** |
| **Input (write)** | `ticket_id`, `summary`, `category`, `resolution_type`, `account_id`, **`customer_id`** |
| **Output** | List of memory records (read); write confirmation (write) |
| **DBs accessed** | `udahub.db` |
| **Used by** | `supervisor_agent` via `langchain-mcp-adapters` |
| **Scoping** | All reads/writes filtered by customer_id |

## Folder Structure

```
solution/
├── agentic/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classifier_agent.py
│   │   ├── resolver_agent.py
│   │   ├── escalation_agent.py
│   │   ├── support_tools_agent.py
│   │   └── supervisor_agent.py
│   ├── design/
│   │   └── architecture.md
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── account_lookup_tool.py
│   │   ├── refund_tool.py
│   │   ├── kb_search_tool.py
│   │   └── memory_tool.py
│   └── workflow.py
├── data/
│   ├── core/
│   ├── external/
│   │   ├── cultpass_articles.jsonl
│   │   ├── cultpass_experiences.jsonl
│   │   └── cultpass_users.jsonl
│   └── models/
│       ├── __init__.py
│       ├── cultpass.py
│       └── udahub.py
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_resolver.py
│   ├── test_tools.py
│   ├── test_workflow.py
│   └── test_kb_search.py
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
├── 03_agentic_app.py
├── utils.py
├── requirements.txt
└── .env.example
```
