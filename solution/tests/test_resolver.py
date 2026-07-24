"""Tests for the Resolver Agent - using mocks for MCP tools and LLM calls."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.agents.resolver_agent import resolver_node


_MOCK_ARTICLES = [
    {
        "article_id": "kb-001",
        "title": "How to Reserve an Event",
        "content": "Open the CultPass app, browse events, select one and tap Reserve.",
        "tags": "onboarding,reserve,event,getting-started",
        "score": 12,
    },
    {
        "article_id": "kb-002",
        "title": "Troubleshooting App Crashes",
        "content": "If the app crashes, try clearing cache or reinstalling.",
        "tags": "technical,crash,app,troubleshoot",
        "score": 8,
    },
]


def _make_mock_llm_response(response_text: str, confidence: float, selected_article: str = ""):
    if selected_article:
        mock_result = MagicMock()
        mock_result.content = f'{{"selected_article": "{selected_article}", "response": "{response_text}", "confidence": {confidence}}}'
    else:
        mock_result = MagicMock()
        mock_result.content = f'{{"response": "{response_text}", "confidence": {confidence}}}'
    return mock_result


def _make_mock_kb_tool(articles=None):
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = articles if articles is not None else []
    return mock_tool


def _run_resolver(ticket_text: str, category: str = "technical", articles=None,
                  llm_response_text: str = "Based on the article, you can reserve events by opening the app.",
                  llm_confidence: float = 0.85) -> dict:
    state = {
        "ticket_id": "test-resolver",
        "ticket_text": ticket_text,
        "user_email": "test@test.com",
        "classification": {"category": category, "urgency": "medium", "routing_label": "resolver"},
        "resolution": {},
        "escalation": {},
        "tool_results": [],
        "memory_context": [],
        "agent_trace": [],
        "messages": [],
    }

    with patch("agentic.agents.resolver_agent._get_kb_search_tools") as mock_get_tools:
        mock_get_tools.return_value = [_make_mock_kb_tool(articles)]

        with patch("agentic.agents.resolver_agent.ChatOpenAI") as mock_chat_cls:
            mock_llm = MagicMock()
            mock_llm.return_value = _make_mock_llm_response(llm_response_text, llm_confidence)
            mock_chat_cls.return_value = mock_llm

            result = resolver_node(state)
    return result


def test_resolver_finds_article():
    result = _run_resolver("How do I reserve a spot for an event?", "onboarding", articles=_MOCK_ARTICLES)
    resolution = result.get("resolution", {})
    assert resolution.get("status") == "resolved", f"Expected resolved, got {resolution.get('status')}"
    assert resolution.get("response", "") != ""
    assert resolution.get("confidence", 0) >= 0.6


def test_resolver_escalates_unmatched():
    result = _run_resolver("My pet elephant escaped during the parade", "billing", articles=[])
    resolution = result.get("resolution", {})
    assert resolution is not None
    assert resolution.get("confidence", 1.0) == 0.0
    assert resolution.get("status") == "failed"


def test_resolver_returns_tool_results():
    result = _run_resolver("App crashes when reserving", "technical", articles=_MOCK_ARTICLES)
    tool_results = result.get("tool_results", [])
    assert len(tool_results) > 0
    tool_names = [t.get("tool") for t in tool_results]
    assert "kb_search" in tool_names


def test_resolver_empty_input():
    result = _run_resolver("")
    resolution = result.get("resolution", {})
    assert resolution.get("status") == "failed"
    assert resolution.get("confidence", 1.0) == 0.0


def test_resolver_low_confidence_escalates():
    result = _run_resolver(
        "I need a completely unrelated thing", "billing",
        articles=_MOCK_ARTICLES,
        llm_response_text="This article does not really address the issue.",
        llm_confidence=0.3,
    )
    resolution = result.get("resolution", {})
    assert resolution.get("status") == "failed"
    assert resolution.get("confidence", 1.0) < 0.6


def test_resolver_includes_memory_context():
    state = {
        "ticket_id": "test-mem",
        "ticket_text": "How do I reserve an event?",
        "user_email": "test@test.com",
        "classification": {"category": "onboarding", "urgency": "low", "routing_label": "resolver"},
        "resolution": {},
        "escalation": {},
        "tool_results": [],
        "memory_context": [
            {"summary": "Previously helped with app setup", "resolution_type": "resolved"},
        ],
        "agent_trace": [],
        "messages": [],
    }

    with patch("agentic.agents.resolver_agent._get_kb_search_tools") as mock_get_tools:
        mock_get_tools.return_value = [_make_mock_kb_tool(_MOCK_ARTICLES)]

        with patch("agentic.agents.resolver_agent.ChatOpenAI") as mock_chat_cls:
            mock_llm = MagicMock()
            mock_llm.return_value = _make_mock_llm_response("Reserve events via the app.", 0.9)
            mock_chat_cls.return_value = mock_llm

            result = resolver_node(state)

    resolution = result.get("resolution", {})
    assert resolution.get("confidence", 0) >= 0.6


if __name__ == "__main__":
    test_resolver_finds_article()
    test_resolver_escalates_unmatched()
    test_resolver_returns_tool_results()
    test_resolver_empty_input()
    test_resolver_low_confidence_escalates()
    test_resolver_includes_memory_context()
    print("All resolver tests passed!")
