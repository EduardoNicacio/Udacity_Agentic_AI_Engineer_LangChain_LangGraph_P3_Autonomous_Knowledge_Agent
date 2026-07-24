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
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools


_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_KB_SEARCH_SERVER_PATH = str(_BASE_DIR / "agentic" / "tools" / "kb_search_tool.py")


def _load_kb_search_tools() -> list:
    """Load kb_search MCP tool via langchain-mcp-adapters."""
    async def _load():
        tools = await load_mcp_tools({
            "kb_search": {
                "transport": "stdio",
                "command": "python",
                "args": [_KB_SEARCH_SERVER_PATH],
            }
        })
        return tools

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _load())
            return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(_load())


_KB_SEARCH_TOOLS = None


def _get_kb_search_tools() -> list:
    global _KB_SEARCH_TOOLS
    if _KB_SEARCH_TOOLS is None:
        try:
            _KB_SEARCH_TOOLS = _load_kb_search_tools()
        except Exception:
            _KB_SEARCH_TOOLS = []
    return _KB_SEARCH_TOOLS


RESOLVER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support agent for CultPass. You have been given a knowledge base article "
        "and a customer ticket. Use the article to answer the customer's question.\n\n"
        "Article title: {article_title}\n"
        "Article content: {article_content}\n\n"
        "Evaluate how well this article addresses the customer's issue. "
        "Consider:\n"
        "1. Does this article directly address the issue? (0.0-0.4)\n"
        "2. Does the article provide enough information to resolve? (0.0-0.3)\n"
        "3. Is the information specific enough for this scenario? (0.0-0.3)\n\n"
        "Respond with a JSON object:\n"
        "{{\n"
        '  "response": "your answer based on the article",\n'
        '  "confidence": 0.85\n'
        "}}\n\n"
        "Confidence should be a float between 0.0 and 1.0."
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

    tool_results = state.get("tool_results", [])
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

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = RESOLVER_PROMPT | llm
        result = chain.invoke({
            "article_title": best.get("title", ""),
            "article_content": best.get("content", ""),
            "ticket_text": ticket_text + (f"\nCustomer history: {mem_context}" if mem_context else ""),
        })

        try:
            parsed = json.loads(result.content.strip().strip("`").replace("json\n", ""))
            response = parsed.get("response", result.content)
            confidence = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            response = result.content
            confidence = 0.5
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
