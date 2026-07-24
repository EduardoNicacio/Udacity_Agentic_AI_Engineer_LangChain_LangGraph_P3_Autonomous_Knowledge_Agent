"""Resolver Agent - performs RAG against the knowledge base and computes confidence score.

Role:
    Search the Knowledge table for articles matching the ticket. Select the best match,
    generate a response from it, and assign a confidence score. If confidence < 0.6,
    the ticket will be escalated by the conditional edge in workflow.py.

Node function:
    resolver_node(state) -> dict  (partial state update)

Input (from state):
    - ticket_text: str
    - classification: dict {category, urgency, routing_label}
    - memory_context: list[dict]

Output (on state):
    - resolution: dict {status, article_id, article_title, response, confidence}
    - tool_results: list[dict] (appended)
"""

import asyncio
import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from utils import log_agent_step


_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_KB_SEARCH_SERVER_PATH = str(_BASE_DIR / "agentic" / "tools" / "kb_search_tool.py")


def _load_kb_search_tools() -> list:
    """Load kb_search MCP tool via langchain-mcp-adapters."""
    async def _load():
        env = os.environ.copy()
        env['PYTHONPATH'] = str(_BASE_DIR)
        client = MultiServerMCPClient({
            "kb_search": StdioConnection(
                transport="stdio",
                command="python",
                args=[_KB_SEARCH_SERVER_PATH],
                cwd=str(_BASE_DIR),
                env=env,
            ),
        })
        return await client.get_tools()

    import threading
    result = [None]
    exception = [None]

    def _target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(_load())
            loop.close()
        except Exception as e:
            exception[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=30)

    if exception[0]:
        raise exception[0]
    return result[0] if result[0] is not None else []


_KB_SEARCH_TOOLS = None


def _get_kb_search_tools() -> list:
    global _KB_SEARCH_TOOLS
    if _KB_SEARCH_TOOLS is None:
        try:
            _KB_SEARCH_TOOLS = _load_kb_search_tools()
        except Exception as e:
            try:
                from agentic.tools.direct_tools import get_kb_search_tool
                _KB_SEARCH_TOOLS = [_DirectTool("kb_search", get_kb_search_tool())]
            except Exception:
                import traceback
                traceback.print_exc()
                _KB_SEARCH_TOOLS = []
    return _KB_SEARCH_TOOLS


class _DirectTool:
    """Wrapper to make a direct function callable like an MCP tool."""
    def __init__(self, name, func):
        self.name = name
        self._func = func
    def invoke(self, input):
        return self._func(**input)


RESOLVER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support agent for CultPass. You have been given a set of knowledge base "
        "articles and a customer ticket. Your job is to:\n"
        "1. Read all articles carefully.\n"
        "2. Pick the ONE article that BEST addresses the customer's specific issue.\n"
        "3. Use that article to compose a helpful response.\n"
        "4. Assign a confidence score based on how well that chosen article resolves the issue.\n\n"
        "Available articles:\n"
        "{articles_block}\n\n"
        "Evaluate the chosen article:\n"
        "1. Does it directly address the issue? (0.0-0.4)\n"
        "2. Does it provide enough information to resolve? (0.0-0.3)\n"
        "3. Is the information specific enough for this scenario? (0.0-0.3)\n\n"
        "Respond with a JSON object:\n"
        "{{\n"
        '  "selected_article": "exact title of the article you chose",\n'
        '  "response": "your answer based on the article",\n'
        '  "confidence": 0.85\n'
        "}}\n\n"
        "Confidence should be a float between 0.0 and 1.0. Be honest — if no article is a good match, assign low confidence."
    ),
    ("human", "Customer ticket: {ticket_text}"),
])


def resolver_node(state: dict) -> dict:
    """Search KB, select best article, and attempt resolution with confidence scoring."""
    ticket_text = state.get("ticket_text", "")
    classification = state.get("classification", {})
    category = classification.get("category", "unknown") if classification else "unknown"
    memory = state.get("memory_context", [])

    trace = state.get("agent_trace", [])
    result_trace = trace + ["resolver_node"]

    if not ticket_text:
        return {
            "resolution": {
                "status": "failed",
                "article_id": "",
                "article_title": "",
                "response": "No ticket text provided.",
                "confidence": 0.0,
            },
            "agent_trace": result_trace,
        }

    matches = []
    kb_tools = _get_kb_search_tools()
    if kb_tools:
        try:
            kb_tool = kb_tools[0]
            matches = kb_tool.invoke({"ticket_text": ticket_text, "category": category})
            if not isinstance(matches, list):
                matches = []
        except Exception:
            matches = []

    tool_results = list(state.get("tool_results", []))
    tool_results.append({
        "tool": "kb_search",
        "matches_found": len(matches),
    })

    if not matches:
        return {
            "resolution": {
                "status": "failed",
                "article_id": "",
                "article_title": "",
                "response": "I could not find a matching article for your issue. Escalating to support.",
                "confidence": 0.0,
            },
            "tool_results": tool_results,
            "agent_trace": result_trace,
        }

    best = matches[0]
    tool_results.append({
        "tool": "kb_best_match",
        "article_id": best.get("article_id", ""),
        "article_title": best.get("title", ""),
        "match_score": best.get("score", 0),
    })

    if memory:
        mem_context = "; ".join([m.get("summary", "") for m in memory[:3]])
    else:
        mem_context = ""

    articles_block = ""
    for i, m in enumerate(matches[:3], 1):
        articles_block += (
            f"Article {i}:\n"
            f"  Title: {m.get('title', '')}\n"
            f"  Content: {m.get('content', '')}\n"
            f"  Tags: {m.get('tags', '')}\n\n"
        )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = RESOLVER_PROMPT | llm
        result = chain.invoke({
            "articles_block": articles_block,
            "ticket_text": ticket_text + (f"\nCustomer history: {mem_context}" if mem_context else ""),
        })

        try:
            parsed = json.loads(result.content.strip().strip("`").replace("json\n", ""))
            response = parsed.get("response", result.content)
            confidence = float(parsed.get("confidence", 0.5))
            selected_title = parsed.get("selected_article", "")
        except (json.JSONDecodeError, ValueError):
            response = result.content
            confidence = 0.5
            selected_title = ""

        if selected_title:
            for m in matches:
                if m.get("title", "").lower() == selected_title.lower():
                    best = m
                    break
    except Exception:
        return {
            "resolution": {
                "status": "failed",
                "article_id": best.get("article_id", ""),
                "article_title": best.get("title", ""),
                "response": "An error occurred while generating a response. Escalating.",
                "confidence": 0.0,
            },
            "tool_results": tool_results,
            "agent_trace": result_trace,
        }

    log_agent_step(
        agent="resolver_node",
        action="kb_search_resolved",
        details={
            "article_id": best.get("article_id", ""),
            "article_title": best.get("title", ""),
            "confidence": confidence,
            "matches_found": len(matches),
        },
        ticket_id=state.get("ticket_id", ""),
    )

    return {
        "resolution": {
            "status": "resolved" if confidence >= 0.6 else "failed",
            "article_id": best.get("article_id", ""),
            "article_title": best.get("title", ""),
            "response": response,
            "confidence": confidence,
        },
        "tool_results": tool_results,
        "agent_trace": result_trace,
    }
