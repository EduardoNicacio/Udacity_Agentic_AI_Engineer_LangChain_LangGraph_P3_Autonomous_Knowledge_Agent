"""Supervisor Agent - entry point and final node for the UDA-Hub workflow.

Role:
    Two node functions:
    - supervisor_node: Entry point. Initializes state, loads long-term memory via
      the memory MCP tool.
    - supervisor_final_node: Exit point. Composes the final assistant response from
      resolution or escalation data. Writes long-term memory record via MCP tool.

Node functions:
    supervisor_node(state) -> dict
    supervisor_final_node(state) -> dict
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from utils import log_agent_step


import os
_MEMORY_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "memory_tool.py")
_MEMORY_SERVER_PATH = os.path.normpath(_MEMORY_SERVER_PATH)


def _load_memory_tools() -> list:
    """Load memory MCP tools via langchain-mcp-adapters."""
    async def _load():
        tools = await load_mcp_tools({
            "memory_tool": {
                "transport": "stdio",
                "command": "python",
                "args": [_MEMORY_SERVER_PATH],
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


_MEMORY_TOOLS = None


def _get_memory_tools() -> list:
    global _MEMORY_TOOLS
    if _MEMORY_TOOLS is None:
        try:
            _MEMORY_TOOLS = _load_memory_tools()
        except Exception:
            _MEMORY_TOOLS = []
    return _MEMORY_TOOLS


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a CultPass customer support assistant. Compose a helpful, friendly response "
        "to the customer based on the resolution or escalation information provided."
    ),
    ("human", "Ticket: {ticket_text}\n\nResolution: {resolution_info}\n\nEscalation: {escalation_info}\n\n"
              "Customer history: {customer_history}"),
])


def supervisor_node(state: dict) -> dict:
    """Entry point: load long-term memory and prepare state."""
    ticket_text = state.get("ticket_text", "")
    ticket_id = state.get("ticket_id", "")

    classification = state.get("classification", {})
    category = classification.get("category", "") if classification else ""

    memories = []
    memory_tools = _get_memory_tools()
    if memory_tools:
        try:
            read_tool = memory_tools[0]
            memories = read_tool.invoke({
                "category": category or "unknown",
                "ticket_text": ticket_text,
                "limit": 5,
            })
            if not isinstance(memories, list):
                memories = []
        except Exception:
            memories = []

    log_agent_step(
        agent="supervisor_node",
        action="ticket_received",
        details={
            "ticket_id": ticket_id,
            "memory_count": len(memories),
        },
        ticket_id=ticket_id,
    )

    return {
        "memory_context": memories,
        "agent_trace": ["supervisor_node"],
    }


def supervisor_final_node(state: dict) -> dict:
    """Exit point: compose final response and write memory."""
    ticket_text = state.get("ticket_text", "")
    ticket_id = state.get("ticket_id", "")
    resolution = state.get("resolution", {})
    escalation = state.get("escalation", {})
    classification = state.get("classification", {})
    memory = state.get("memory_context", [])

    category = classification.get("category", "unknown") if classification else "unknown"

    if resolution and resolution.get("status") == "resolved":
        resolution_info = f"Resolved with confidence {resolution.get('confidence', 0)}: {resolution.get('response', '')}"
        escalation_info = "None needed"
        resolution_type = "resolved"
        summary = f"Resolved: {resolution.get('response', '')[:200]}"
    else:
        resolution_info = resolution.get("response", "") if resolution else ""
        escalation_info = f"Escalated: {escalation.get('reason', '')}" if escalation else "None"
        resolution_type = "escalated"
        summary = f"Escalated: {escalation.get('reason', '')[:200]}" if escalation else "Escalated"

    customer_history = "; ".join([m.get("summary", "") for m in memory[:3]]) if memory else ""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = FINAL_PROMPT | llm
        result = chain.invoke({
            "ticket_text": ticket_text,
            "resolution_info": resolution_info,
            "escalation_info": escalation_info,
            "customer_history": customer_history or "No prior history",
        })
        response = result.content
    except Exception:
        if resolution and resolution.get("status") == "resolved":
            response = resolution.get("response", "Thank you for contacting us.")
        else:
            response = "Your issue has been escalated to our support team. They will follow up soon."

    if ticket_id:
        memory_tools = _get_memory_tools()
        if memory_tools and len(memory_tools) > 1:
            try:
                write_tool = memory_tools[1]
                write_tool.invoke({
                    "ticket_id": ticket_id,
                    "summary": summary,
                    "category": category,
                    "resolution_type": resolution_type,
                })
            except Exception:
                pass

    log_agent_step(
        agent="supervisor_final_node",
        action="session_complete",
        details={
            "resolution_type": resolution_type,
            "category": category,
        },
        ticket_id=ticket_id,
    )

    return {
        "messages": [AIMessage(content=response)],
        "agent_trace": state.get("agent_trace", []) + ["supervisor_final_node"],
    }
