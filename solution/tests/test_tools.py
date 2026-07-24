"""Tests for the MCP-based tools - account_lookup and refund_tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.tools.account_lookup_tool import account_lookup
from agentic.tools.refund_tool import process_refund


def test_account_lookup_by_email():
    """Look up a known user by email."""
    result = account_lookup(email="alice.kingsley@wonderland.com")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("name") == "Alice Kingsley"
    assert result.get("user_id") == "a4ab87"
    assert "subscription_tier" in result
    assert "subscription_status" in result


def test_account_lookup_by_id():
    """Look up a known user by user_id."""
    result = account_lookup(user_id="f556c0")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("name") == "Bob Stone"


def test_account_lookup_not_found():
    """Look up a nonexistent user should return an error."""
    result = account_lookup(email="nonexistent@nowhere.com")
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_account_lookup_no_input():
    """No input should return error."""
    result = account_lookup()
    assert "error" in result


def test_refund_invalid_ticket():
    """Refund for nonexistent ticket should return error."""
    result = process_refund(ticket_id="fake-ticket-999", amount=29.99)
    assert "error" in result


def test_refund_negative_amount():
    """Refund with negative amount should fail."""
    result = process_refund(ticket_id="fake-id", amount=-10.0)
    assert "error" in result
    assert "positive" in result["error"].lower()


if __name__ == "__main__":
    test_account_lookup_by_email()
    test_account_lookup_by_id()
    test_account_lookup_not_found()
    test_account_lookup_no_input()
    test_refund_invalid_ticket()
    test_refund_negative_amount()
    print("All tool tests passed!")
