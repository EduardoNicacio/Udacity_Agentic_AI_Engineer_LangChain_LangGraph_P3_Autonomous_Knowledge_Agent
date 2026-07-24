"""FastMCP server for looking up CultPass account information.

Input: email (str) OR user_id (str)
Output: {user_id, name, email, subscription_tier, subscription_status, monthly_quota, open_ticket_count}
"""

from pathlib import Path
from uuid import uuid4
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models import cultpass as cultpass_models
from data.models import udahub as udahub_models

mcp = FastMCP("account_lookup")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _get_engines():
    cultpass_db = str(_BASE_DIR / "data" / "external" / "cultpass.db")
    udahub_db = str(_BASE_DIR / "data" / "core" / "udahub.db")
    cultpass_engine = create_engine(f"sqlite:///{cultpass_db}", echo=False)
    udahub_engine = create_engine(f"sqlite:///{udahub_db}", echo=False)
    return cultpass_engine, udahub_engine


@mcp.tool()
def account_lookup(email: str = "", user_id: str = "") -> dict:
    """Look up a CultPass account by email or user_id.

    Args:
        email: The user's email address.
        user_id: The user's external ID.

    Returns:
        A dict with user info, subscription status, and open ticket count.
    """
    if not email and not user_id:
        return {"error": "Provide at least one of email or user_id"}

    cultpass_engine, udahub_engine = _get_engines()

    with sessionmaker(bind=cultpass_engine)() as cultpass_session:
        query = cultpass_session.query(cultpass_models.User)
        if email:
            user = query.filter(cultpass_models.User.email == email).first()
        else:
            user = query.filter(cultpass_models.User.user_id == user_id).first()

        if not user:
            return {"error": f"User not found"}

        subscription = cultpass_session.query(cultpass_models.Subscription).filter(
            cultpass_models.Subscription.user_id == user.user_id
        ).first()

        result = {
            "user_id": user.user_id,
            "name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "subscription_tier": "",
            "subscription_status": "",
            "monthly_quota": 0,
            "open_ticket_count": 0,
        }

        if subscription:
            result["subscription_tier"] = subscription.tier
            result["subscription_status"] = subscription.status
            result["monthly_quota"] = subscription.monthly_quota

    with sessionmaker(bind=udahub_engine)() as udahub_session:
        udahub_user = udahub_session.query(udahub_models.User).filter(
            udahub_models.User.external_user_id == user.user_id
        ).first()

        if udahub_user:
            open_tickets = udahub_session.query(udahub_models.Ticket).join(
                udahub_models.TicketMetadata
            ).filter(
                udahub_models.Ticket.user_id == udahub_user.user_id,
                udahub_models.TicketMetadata.status == "open",
            ).count()
            result["open_ticket_count"] = open_tickets

    return result


if __name__ == "__main__":
    mcp.run()
