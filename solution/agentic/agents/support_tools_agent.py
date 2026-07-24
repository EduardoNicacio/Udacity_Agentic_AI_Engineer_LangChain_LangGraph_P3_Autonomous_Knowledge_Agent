"""Support Tools Agent - invokes account_lookup and refund MCP tools when applicable.

Role:
    After classification, this node checks whether billing/account/refund tools
    should be invoked based on the ticket category and content. Results are
    recorded in state.tool_results for downstream agents to use.

Node function:
    support_tools_node(state) -> dict
"""

import asyncio
import os
from langchain_mcp_adapters.tools import load_mcp_tools
from utils import log_agent_step


_ACCOUNT_LOOKUP_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "account_lookup_tool.py")
_REFUND_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "refund_tool.py")
_ACCOUNT_LOOKUP_PATH = os.path.normpath(_ACCOUNT_LOOKUP_PATH)
_REFUND_PATH = os.path.normpath(_REFUND_PATH)


def _load_support_tools() -> list:
    """Load account_lookup and refund MCP tools via langchain-mcp-adapters."""
    async def _load():
        tools = await load_mcp_tools({
            "account_lookup_tool": {
                "transport": "stdio",
                "command": "python",
                "args": [_ACCOUNT_LOOKUP_PATH],
            },
            "refund_tool": {
                "transport": "stdio",
                "command": "python",
                "args": [_REFUND_PATH],
            },
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


_SUPPORT_TOOLS = None


def _get_support_tools() -> list:
    global _SUPPORT_TOOLS
    if _SUPPORT_TOOLS is None:
        try:
            _SUPPORT_TOOLS = _load_support_tools()
        except Exception:
            _SUPPORT_TOOLS = []
    return _SUPPORT_TOOLS


def support_tools_node(state: dict) -> dict:
    """Invoke support tools based on ticket classification and content.

    For billing/account tickets: invoke account_lookup with user_email.
    For refund-related tickets: invoke process_refund.
    Results are appended to tool_results.
    """
    ticket_text = state.get("ticket_text", "")
    ticket_id = state.get("ticket_id", "")
    user_email = state.get("user_email", "")
    classification = state.get("classification", {})
    category = classification.get("category", "unknown") if classification else "unknown"
    ticket_lower = ticket_text.lower()

    tool_results = list(state.get("tool_results", []))
    support_tools = _get_support_tools()

    if not support_tools:
        log_agent_step(
            agent="support_tools_node",
            action="tools_unavailable",
            details={"category": category},
            ticket_id=ticket_id,
        )
        return {"tool_results": tool_results}

    tool_map = {}
    for t in support_tools:
        tool_map[t.name] = t

    invoked = []

    if category == "billing" and user_email:
        if "account_lookup" in tool_map:
            try:
                result = tool_map["account_lookup"].invoke({"email": user_email})
                tool_results.append({
                    "tool": "account_lookup",
                    "input": {"email": user_email},
                    "result": result,
                })
                invoked.append("account_lookup")
            except Exception as e:
                tool_results.append({
                    "tool": "account_lookup",
                    "input": {"email": user_email},
                    "result": {"error": str(e)},
                })

    if "refund" in ticket_lower or ("billing" in ticket_lower and "refund" in ticket_lower):
        if "process_refund" in tool_map:
            try:
                refund_input = {"ticket_id": ticket_id, "reason": ticket_text[:500]}
                result = tool_map["process_refund"].invoke(refund_input)
                tool_results.append({
                    "tool": "process_refund",
                    "input": refund_input,
                    "result": result,
                })
                invoked.append("process_refund")
            except Exception as e:
                tool_results.append({
                    "tool": "process_refund",
                    "input": {"ticket_id": ticket_id},
                    "result": {"error": str(e)},
                })

    trace = state.get("agent_trace", [])

    log_agent_step(
        agent="support_tools_node",
        action="support_tools_invoked",
        details={
            "invoked": invoked,
            "category": category,
            "tool_results_count": len(tool_results),
        },
        ticket_id=ticket_id,
    )

    return {"tool_results": tool_results, "agent_trace": trace + ["support_tools_node"]}
