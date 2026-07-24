"""Shared utilities for UDA-Hub: DB helpers, structured logger, memory utilities, ticket parser."""

import json
import os
from datetime import datetime, timezone
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()


def reset_db(db_path: str, echo: bool = True):
    """Drop the existing SQLite DB file and recreate all tables."""
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)


@contextmanager
def get_session(engine: Engine):
    """Context manager for a SQLAlchemy session with auto-commit/rollback."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def log_agent_step(agent: str, action: str, details: dict, ticket_id: str = ""):
    """Emit a structured JSON log line for an agent step.

    Args:
        agent: Name of the agent or node.
        action: Action being performed (e.g., classify_ticket, kb_search).
        details: Dict with action-specific details.
        ticket_id: Optional ticket identifier for correlation.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "agent": agent,
        "action": action,
        "ticket_id": ticket_id,
        "details": details,
    }
    print(json.dumps(log_entry, default=str))


def parse_ticket_input(raw_text: str) -> dict:
    """Parse raw user input into ticket_text and optionally extract user_email.

    Args:
        raw_text: The raw input string from the user.

    Returns:
        dict with keys ticket_text and user_email.
    """
    import re
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(email_pattern, raw_text)

    ticket_text = raw_text.strip()
    user_email = emails[0] if emails else "unknown@email.com"

    return {
        "ticket_text": ticket_text,
        "user_email": user_email,
    }


def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    """Interactive chat loop using the compiled agent graph.

    Preserves conversation state across turns using the thread_id checkpointer.
    Each turn appends new messages to the existing state rather than resetting it.

    Args:
        agent: Compiled LangGraph state graph.
        ticket_id: Thread/ticket identifier for checkpointing.

    Usage:
        from utils import chat_interface
        from agentic.workflow import compile_graph
        graph = compile_graph()
        chat_interface(graph, "ticket-123")
    """
    print(f"\n--- UDA-Hub Customer Support Agent ---")
    print(f"Session: {ticket_id}")
    print("Type your ticket or question. Type 'quit', 'exit', or 'q' to end.\n")

    thread_counter = 0
    current_thread_id = ticket_id

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break

        parsed = parse_ticket_input(user_input)
        thread_counter += 1
        current_thread_id = f"{ticket_id}-{thread_counter}"

        # For multi-turn: on first turn, provide full initial state;
        # on subsequent turns, the checkpointer (via thread_id) preserves prior state.
        # We only need to supply the new message and ticket fields.
        state_update = {
            "ticket_id": current_thread_id,
            "ticket_text": parsed["ticket_text"],
            "user_email": parsed["user_email"],
            "customer_id": parsed["user_email"],
            "classification": {},
            "resolution": {},
            "escalation": {},
            "tool_results": [],
            "memory_context": [],
            "agent_trace": [],
            "messages": [HumanMessage(content=parsed["ticket_text"])],
        }

        config = {
            "configurable": {
                "thread_id": current_thread_id,
            }
        }

        result = agent.invoke(input=state_update, config=config)

        messages = result.get("messages", [])
        agent_trace = result.get("agent_trace", [])

        if messages:
            last_msg = messages[-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            print(f"Assistant: {content}")
        else:
            print("Assistant: No response generated.")

        resolution = result.get("resolution", {})
        escalation = result.get("escalation", {})

        if resolution and resolution.get("status") == "resolved":
            print(f"[Resolved - confidence: {resolution.get('confidence', 0):.2f}]")
        elif escalation and escalation.get("reason"):
            print(f"[Escalated - priority: {escalation.get('priority', 'unknown')}]")

        trace_str = " -> ".join(agent_trace) if agent_trace else "(trace not recorded)"
        print(f"[Trace: {trace_str}]\n")
