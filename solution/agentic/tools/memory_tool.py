"""FastMCP server for reading and writing long-term conversation memory.

Reads and writes ConversationMemory records in the UDA-Hub core database.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models import udahub as udahub_models

mcp = FastMCP("memory_tool")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_STOP_WORDS = {"the", "a", "an", "is", "was", "to", "for", "of", "in", "it", "on", "at", "my", "i", "me"}


@mcp.tool()
def read_memory(category: str, ticket_text: str, limit: int = 5, account_id: str = "cultpass", customer_id: str = "") -> list[dict]:
    """Read past conversation memory by category and keyword overlap, scoped to a customer.

    Args:
        category: The ticket category to filter memories by.
        ticket_text: The current ticket text for keyword matching.
        limit: Maximum number of memories to return.
        account_id: The account ID to scope the search to.
        customer_id: The customer (external user) ID to scope memory to.

    Returns:
        A list of matching memory records.
    """
    keywords = [w.lower().strip(".,!?;:'\"") for w in ticket_text.split()
                if w.lower().strip(".,!?;:'\"") not in _STOP_WORDS and len(w) > 2]

    udahub_db = str(_BASE_DIR / "data" / "core" / "udahub.db")
    engine = create_engine(f"sqlite:///{udahub_db}", echo=False)

    memories = []
    try:
        with sessionmaker(bind=engine)() as session:
            base_filters = [
                udahub_models.ConversationMemory.account_id == account_id,
            ]
            if customer_id:
                base_filters.append(udahub_models.ConversationMemory.customer_id == customer_id)

            if category and category != "unknown":
                mems = session.query(udahub_models.ConversationMemory).filter(
                    *base_filters,
                    udahub_models.ConversationMemory.category == category,
                ).order_by(
                    udahub_models.ConversationMemory.created_at.desc()
                ).limit(limit).all()
                for m in mems:
                    memories.append({
                        "memory_id": m.memory_id,
                        "customer_id": m.customer_id,
                        "summary": m.summary,
                        "category": m.category,
                        "resolution_type": m.resolution_type,
                        "created_at": str(m.created_at),
                    })

            if not memories:
                for kw in keywords:
                    mems = session.query(udahub_models.ConversationMemory).filter(
                        *base_filters,
                        udahub_models.ConversationMemory.summary.like(f"%{kw}%"),
                    ).order_by(
                        udahub_models.ConversationMemory.created_at.desc()
                    ).limit(limit).all()
                    if mems:
                        for m in mems:
                            memories.append({
                                "memory_id": m.memory_id,
                                "customer_id": m.customer_id,
                                "summary": m.summary,
                                "category": m.category,
                                "resolution_type": m.resolution_type,
                                "created_at": str(m.created_at),
                            })
                        break
    except Exception:
        pass

    return memories[:limit]


@mcp.tool()
def write_memory(
    ticket_id: str,
    summary: str,
    category: str,
    resolution_type: str,
    account_id: str = "cultpass",
    customer_id: str = "",
) -> dict:
    """Write a conversation memory record after resolution or escalation.

    Args:
        ticket_id: The ticket ID to associate with this memory.
        summary: A summary of the resolution or escalation.
        category: The ticket category.
        resolution_type: Either 'resolved' or 'escalated'.
        account_id: The account ID.
        customer_id: The customer (external user) ID.

    Returns:
        A dict confirming the write operation.
    """
    udahub_db = str(_BASE_DIR / "data" / "core" / "udahub.db")
    engine = create_engine(f"sqlite:///{udahub_db}", echo=False)
    try:
        with sessionmaker(bind=engine)() as session:
            memory = udahub_models.ConversationMemory(
                memory_id=str(uuid4()),
                account_id=account_id,
                customer_id=customer_id,
                ticket_id=ticket_id,
                summary=summary,
                category=category,
                resolution_type=resolution_type,
                created_at=datetime.now(timezone.utc),
            )
            session.add(memory)
            session.commit()
            return {"status": "written", "memory_id": memory.memory_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    mcp.run()
