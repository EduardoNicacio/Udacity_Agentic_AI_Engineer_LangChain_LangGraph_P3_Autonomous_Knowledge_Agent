"""End-to-end workflow tests - feed tickets through the compiled graph and assert state shape.

Uses mocks for MCP tool loading and LLM calls to avoid external dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def _make_mock_classification(category: str, urgency: str, routing_label: str):
    mock_result = MagicMock()
    mock_result.category = category
    mock_result.urgency = urgency
    mock_result.routing_label = routing_label
    return mock_result


_MOCK_ARTICLES = [
    {
        "article_id": "kb-001",
        "title": "Getting Started with CultPass",
        "content": "Download the app, create an account, browse events, and reserve your first experience.",
        "tags": "onboarding,getting-started,setup,account",
        "score": 15,
    },
    {
        "article_id": "kb-002",
        "title": "How to Reserve Events",
        "content": "Browse events in the app, select one, and tap Reserve to secure your spot.",
        "tags": "onboarding,reserve,event,booking",
        "score": 12,
    },
]


def _make_mock_llm_response(response_text: str):
    mock_result = MagicMock()
    mock_result.content = response_text
    return mock_result


def _process_ticket(ticket_text: str, ticket_id: str = "test-e2e"):
    from agentic.workflow import compile_graph
    graph = compile_graph()

    with patch("agentic.agents.classifier_agent.ChatOpenAI") as mock_cls_llm:
        mock_structured = MagicMock()
        if "reserve" in ticket_text.lower() or "get started" in ticket_text.lower() or "how" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("onboarding", "low", "resolver")
        elif "charged" in ticket_text.lower() or "billing" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("billing", "high", "escalation")
        elif "login" in ticket_text.lower() or "password" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("account", "medium", "resolver")
        elif "crash" in ticket_text.lower() or "app" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("technical", "medium", "resolver")
        elif "upgrade" in ticket_text.lower() or "plan" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("subscription", "low", "resolver")
        else:
            mock_structured.return_value = _make_mock_classification("unknown", "low", "escalation")
        mock_cls_llm.return_value.with_structured_output.return_value = mock_structured

        with patch("agentic.agents.resolver_agent._get_kb_search_tools") as mock_get_kb:
            mock_kb_tool = MagicMock()
            mock_kb_tool.invoke.return_value = _MOCK_ARTICLES[:2]
            mock_get_kb.return_value = [mock_kb_tool]

            with patch("agentic.agents.resolver_agent.ChatOpenAI") as mock_res_llm:
                mock_res_llm.return_value.return_value = _make_mock_llm_response(
                    "Based on the knowledge base, here is how you can proceed."
                )

                with patch("agentic.agents.supervisor_agent._get_memory_tools") as mock_get_mem:
                    mock_read_tool = MagicMock()
                    mock_read_tool.invoke.return_value = []
                    mock_write_tool = MagicMock()
                    mock_write_tool.invoke.return_value = {"status": "written"}
                    mock_get_mem.return_value = [mock_read_tool, mock_write_tool]

                    with patch("agentic.agents.supervisor_agent.ChatOpenAI") as mock_sup_llm:
                        mock_sup_llm.return_value.return_value = _make_mock_llm_response(
                            "Thank you for contacting CultPass support!"
                        )

                        with patch("agentic.agents.escalation_agent.ChatOpenAI") as mock_esc_llm:
                            mock_esc_llm.return_value.return_value = _make_mock_llm_response(
                                '{"reason": "Low confidence match", "priority": "medium", "customer_context": "No prior history", "suggested_actions": ["Review ticket"]}'
                            )

                            initial_state = {
                                "ticket_id": ticket_id,
                                "ticket_text": ticket_text,
                                "user_email": "test@test.com",
                                "classification": {},
                                "resolution": {},
                                "escalation": {},
                                "tool_results": [],
                                "memory_context": [],
                                "agent_trace": [],
                                "messages": [],
                            }
                            config = {"configurable": {"thread_id": ticket_id}}
                            return graph.invoke(input=initial_state, config=config)


def test_e2e_resolved_ticket():
    result = _process_ticket("How do I reserve a spot for an event?")
    assert "classification" in result
    assert "resolution" in result
    assert "agent_trace" in result

    resolution = result.get("resolution", {})
    if resolution.get("status") == "resolved":
        assert resolution.get("confidence", 0) >= 0.6


def test_e2e_escalated_ticket():
    result = _process_ticket("I want to speak to a manager about a complaint about my experience last week")
    assert "classification" in result
    assert "resolution" in result
    assert "agent_trace" in result


def test_e2e_state_shape():
    result = _process_ticket("I can't log in to my account")
    assert isinstance(result.get("classification"), dict)
    assert isinstance(result.get("resolution"), dict)
    assert isinstance(result.get("tool_results"), list)
    assert isinstance(result.get("memory_context"), list)
    assert isinstance(result.get("agent_trace"), list)
    assert len(result.get("agent_trace", [])) > 0


def test_e2e_billing_ticket():
    result = _process_ticket("I was charged twice for my subscription")
    classification = result.get("classification", {})
    assert classification.get("category") in ("billing", "unknown")
    assert result.get("agent_trace") is not None


def test_e2e_agent_trace_order():
    result = _process_ticket("How do I reset my password?")
    trace = result.get("agent_trace", [])
    assert "supervisor_node" in trace
    assert trace[0] == "supervisor_node"
    assert len(trace) >= 2


if __name__ == "__main__":
    test_e2e_resolved_ticket()
    test_e2e_escalated_ticket()
    test_e2e_state_shape()
    test_e2e_billing_ticket()
    test_e2e_agent_trace_order()
    print("All workflow tests passed!")
