"""UDA-Hub LangGraph workflow - built from scratch using StateGraph.

This module defines the shared state schema, all agent nodes, conditional routing,
short-term memory (checkpointer), and long-term memory integration.

Exports:
    compile_graph() -> CompiledStateGraph
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

from agentic.agents.classifier_agent import classifier_node
from agentic.agents.resolver_agent import resolver_node
from agentic.agents.escalation_agent import escalation_node
from agentic.agents.supervisor_agent import supervisor_node, supervisor_final_node
from utils import log_agent_step


class AgentState(TypedDict):
    """Shared state schema for the UDA-Hub agent graph.

    Fields:
        ticket_id: Unique identifier for the current ticket.
        ticket_text: Raw customer ticket text.
        user_email: Email address of the customer.
        classification: dict with keys {category, urgency, routing_label}.
        resolution: dict with keys {status, article_id, article_title, response, confidence}.
        escalation: dict with keys {reason, priority, customer_context, suggested_actions}.
        tool_results: List of dicts recording each tool call outcome.
        memory_context: List of dicts from long-term memory retrieval.
        agent_trace: Ordered list of agent names that have run.
        messages: Message history (list of BaseMessage).
    """
    ticket_id: str
    ticket_text: str
    user_email: str
    classification: dict
    resolution: dict
    escalation: dict
    tool_results: list[dict]
    memory_context: list[dict]
    agent_trace: list[str]
    messages: list


def route_after_resolver(state: AgentState) -> str:
    """Conditional edge: decide whether to resolve or escalate based on confidence."""
    resolution = state.get("resolution", {})
    confidence = resolution.get("confidence", 0.0) if resolution else 0.0

    if confidence >= 0.6:
        action = "resolve"
    else:
        action = "escalate"

    log_agent_step(
        agent="router",
        action="routing_decision",
        details={
            "confidence": confidence,
            "threshold": 0.6,
            "decision": action,
        },
        ticket_id=state.get("ticket_id", ""),
    )

    return action


def compile_graph() -> StateGraph:
    """Build and compile the UDA-Hub agent graph.

    Graph structure:
        START -> supervisor -> classifier -> resolver -> (conditional)
            resolve -> supervisor_final -> END
            escalate -> escalation -> supervisor_final -> END

    Returns:
        CompiledStateGraph with MemorySaver checkpointing.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)             # type: ignore
    workflow.add_node("classifier", classifier_node)             # type: ignore
    workflow.add_node("resolver", resolver_node)                 # type: ignore
    workflow.add_node("escalation", escalation_node)             # type: ignore
    workflow.add_node("supervisor_final", supervisor_final_node) # type: ignore

    workflow.set_entry_point("supervisor")

    workflow.add_edge("supervisor", "classifier")
    workflow.add_edge("classifier", "resolver")

    workflow.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {
            "resolve": "supervisor_final",
            "escalate": "escalation",
        },
    )

    workflow.add_edge("escalation", "supervisor_final")
    workflow.add_edge("supervisor_final", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer) # type: ignore
