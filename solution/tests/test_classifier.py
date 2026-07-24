"""Tests for the Classifier Agent - using mocks for LLM calls."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.agents.classifier_agent import classifier_node


def _make_mock_classification(category: str, urgency: str, routing_label: str):
    mock_result = MagicMock()
    mock_result.category = category
    mock_result.urgency = urgency
    mock_result.routing_label = routing_label
    return mock_result


def _make_mock_llm(expected_category, expected_urgency, expected_routing):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.return_value = _make_mock_classification(
        expected_category, expected_urgency, expected_routing
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def _run_classifier(ticket_text: str, user_email: str = "test@test.com") -> dict:
    state = {
        "ticket_id": "test-1",
        "ticket_text": ticket_text,
        "user_email": user_email,
        "classification": {},
        "resolution": {},
        "escalation": {},
        "tool_results": [],
        "memory_context": [],
        "agent_trace": [],
        "messages": [],
    }
    result = classifier_node(state)
    return result.get("classification", {})


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_billing_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("billing", "high", "resolver")
    result = _run_classifier("I was charged twice for my subscription this month")
    assert result.get("category") == "billing", f"Expected billing, got {result.get('category')}"
    assert result.get("urgency") == "high"
    assert result.get("routing_label") == "resolver"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_account_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("account", "medium", "resolver")
    result = _run_classifier("I can't log in to my Cultpass account, forgot my password")
    assert result.get("category") == "account", f"Expected account, got {result.get('category')}"
    assert result.get("urgency") == "medium"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_technical_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("technical", "medium", "resolver")
    result = _run_classifier("The app keeps crashing every time I try to reserve an event")
    assert result.get("category") == "technical", f"Expected technical, got {result.get('category')}"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_subscription_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("subscription", "low", "resolver")
    result = _run_classifier("I want to upgrade from basic to premium plan")
    assert result.get("category") == "subscription", f"Expected subscription, got {result.get('category')}"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_onboarding_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("onboarding", "low", "resolver")
    result = _run_classifier("How do I get started with CultPass?")
    assert result.get("category") == "onboarding", f"Expected onboarding, got {result.get('category')}"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_content_classification(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("content", "low", "resolver")
    result = _run_classifier("How do I leave a review for an experience?")
    assert result.get("category") == "content", f"Expected content, got {result.get('category')}"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_escalation_routing(mock_chat_cls):
    mock_chat_cls.return_value = _make_mock_llm("billing", "high", "escalation")
    result = _run_classifier("I want a full refund and to cancel my account immediately")
    assert result.get("category") == "billing"
    assert result.get("routing_label") == "escalation"


def test_empty_input():
    result = _run_classifier("")
    assert result.get("category") == "unknown"
    assert result.get("routing_label") == "escalation"


@patch("agentic.agents.classifier_agent.ChatOpenAI")
def test_llm_failure_fallback(mock_chat_cls):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.side_effect = Exception("API error")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_cls.return_value = mock_llm

    result = _run_classifier("This should trigger fallback")
    assert result.get("category") == "unknown"
    assert result.get("urgency") == "low"
    assert result.get("routing_label") == "escalation"


if __name__ == "__main__":
    test_billing_classification()
    test_account_classification()
    test_technical_classification()
    test_subscription_classification()
    test_onboarding_classification()
    test_content_classification()
    test_escalation_routing()
    test_empty_input()
    test_llm_failure_fallback()
    print("All classifier tests passed!")
