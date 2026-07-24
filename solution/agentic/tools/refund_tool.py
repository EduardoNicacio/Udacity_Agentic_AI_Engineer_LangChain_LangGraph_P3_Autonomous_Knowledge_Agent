"""FastMCP server for processing refund actions.

Input: ticket_id (str), amount (float), reason (str)
Output: {status, ticket_id, amount, refund_id, reason, processed_at}
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models import udahub as udahub_models

mcp = FastMCP("refund_tool")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


@mcp.tool()
def process_refund(ticket_id: str, amount: float, reason: str = "") -> dict:
    """Process a refund for a given support ticket.

    Args:
        ticket_id: The ID of the support ticket.
        amount: The refund amount.
        reason: Reason for the refund.

    Returns:
        A dict with refund status and details.
    """
    if amount <= 0:
        return {"error": "Refund amount must be positive"}

    udahub_db = str(_BASE_DIR / "data" / "core" / "udahub.db")
    engine = create_engine(f"sqlite:///{udahub_db}", echo=False)

    with sessionmaker(bind=engine)() as session:
        ticket = session.query(udahub_models.Ticket).filter(
            udahub_models.Ticket.ticket_id == ticket_id
        ).first()

        if not ticket:
            return {"error": f"Ticket {ticket_id} not found"}

        metadata = session.query(udahub_models.TicketMetadata).filter(
            udahub_models.TicketMetadata.ticket_id == ticket_id
        ).first()

        if metadata and metadata.status != "open":
            return {"error": f"Ticket {ticket_id} is already closed (status: {metadata.status})"}

        refund_id = str(uuid4())
        now = datetime.now(timezone.utc)

        refund = udahub_models.RefundAction(
            refund_id=refund_id,
            ticket_id=ticket_id,
            amount=amount,
            reason=reason or "No reason provided",
            status="approved",
            processed_at=now,
        )

        session.add(refund)

        if metadata:
            metadata.status = "refunded"

        session.commit()

        return {
            "status": "approved",
            "ticket_id": ticket_id,
            "amount": amount,
            "refund_id": refund_id,
            "reason": reason,
            "processed_at": now.isoformat(),
        }


if __name__ == "__main__":
    mcp.run()
