"""End-to-end workflow tests - feed tickets through the compiled graph and assert state shape.

Uses mocks for MCP tool loading and LLM calls to avoid external dependencies.
Tests prove real resolve/escalate paths, tool usage, and agent trace ordering.
"""

import sys
import io
import json
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
    {
        "article_id": "kb-003",
        "title": "Subscription Billing and Payments",
        "content": "Manage your subscription, update payment method, view invoices.",
        "tags": "billing,subscription,payment,invoice",
        "score": 18,
    },
]


def _make_mock_llm_response(response_text: str, confidence: float = 0.85):
    mock_result = MagicMock()
    mock_result.content = f'{{"response": "{response_text}", "confidence": {confidence}}}'
    return mock_result


def _process_ticket(ticket_text: str, ticket_id: str = "test-e2e", user_email: str = "test@test.com"):
    from agentic.workflow import compile_graph
    graph = compile_graph()

    with patch("agentic.agents.classifier_agent.ChatOpenAI") as mock_cls_llm:
        mock_structured = MagicMock()
        if "reserve" in ticket_text.lower() or "get started" in ticket_text.lower() or "how" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("onboarding", "low", "resolver")
        elif "charged" in ticket_text.lower() or "billing" in ticket_text.lower() or "refund" in ticket_text.lower():
            mock_structured.return_value = _make_mock_classification("billing", "high", "resolver")
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
                    "Based on the knowledge base, here is how you can proceed.", 0.85
                )

                with patch("agentic.agents.supervisor_agent._get_memory_tools") as mock_get_mem:
                    mock_read_tool = MagicMock()
                    mock_read_tool.invoke.return_value = []
                    mock_write_tool = MagicMock()
                    mock_write_tool.invoke.return_value = {"status": "written"}
                    mock_get_mem.return_value = [mock_read_tool, mock_write_tool]

                    with patch("agentic.agents.supervisor_agent.ChatOpenAI") as mock_sup_llm:
                        mock_sup_result = MagicMock()
                        mock_sup_result.content = "Thank you for contacting CultPass support!"
                        mock_sup_llm.return_value.return_value = mock_sup_result

                        with patch("agentic.agents.escalation_agent.ChatOpenAI") as mock_esc_llm:
                            mock_esc_result = MagicMock()
                            mock_esc_result.content = '{"reason": "Low confidence match", "priority": "medium", "customer_context": "No prior history", "suggested_actions": ["Review ticket"]}'
                            mock_esc_llm.return_value.return_value = mock_esc_result

                            with patch("agentic.agents.support_tools_agent._get_support_tools") as mock_get_support:
                                mock_account_tool = MagicMock()
                                mock_account_tool.name = "account_lookup"
                                mock_account_tool.invoke.return_value = {
                                    "email": user_email,
                                    "account_id": "acct-123",
                                    "status": "active",
                                }
                                mock_refund_tool = MagicMock()
                                mock_refund_tool.name = "process_refund"
                                mock_refund_tool.invoke.return_value = {
                                    "ticket_id": ticket_id,
                                    "status": "approved",
                                }
                                mock_get_support.return_value = [mock_account_tool, mock_refund_tool]

                                initial_state = {
                                    "ticket_id": ticket_id,
                                    "ticket_text": ticket_text,
                                    "user_email": user_email,
                                    "customer_id": user_email,
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
    assert resolution.get("status") == "resolved", f"Expected resolved, got: {resolution}"
    assert resolution.get("confidence", 0) >= 0.6, f"Confidence too low: {resolution.get('confidence')}"
    assert resolution.get("article_id"), "Missing article_id"
    assert resolution.get("response"), "Missing response"


def test_e2e_escalated_ticket():
    result = _process_ticket("I want to speak to a manager about a complaint about my experience last week")
    escalation = result.get("escalation", {})
    assert escalation.get("reason"), f"Expected escalation reason, got: {escalation}"
    assert escalation.get("priority") in ("high", "medium", "low"), f"Invalid priority: {escalation.get('priority')}"
    assert result.get("resolution", {}).get("status") != "resolved", "Should not be resolved"


def test_e2e_state_shape():
    result = _process_ticket("I can't log in to my account")
    assert isinstance(result.get("classification"), dict)
    assert isinstance(result.get("resolution"), dict)
    assert isinstance(result.get("tool_results"), list)
    assert isinstance(result.get("memory_context"), list)
    assert isinstance(result.get("agent_trace"), list)
    assert len(result.get("agent_trace", [])) >= 3, "Trace should have at least 3 nodes"


def test_e2e_billing_ticket():
    result = _process_ticket("I was charged twice for my subscription", user_email="billing@test.com")
    classification = result.get("classification", {})
    assert classification.get("category") == "billing", f"Expected billing, got: {classification.get('category')}"
    assert result.get("agent_trace") is not None
    tool_results = result.get("tool_results", [])
    assert len(tool_results) > 0, "Billing ticket should invoke support tools"


def test_e2e_agent_trace_strict_order():
    result = _process_ticket("How do I reset my password?")
    trace = result.get("agent_trace", [])
    assert trace[0] == "supervisor_node", f"First node should be supervisor_node, got: {trace[0]}"
    assert trace[1] == "classifier_node", f"Second node should be classifier_node, got: {trace[1]}"
    assert "support_tools_node" in trace, f"support_tools_node should be in trace: {trace}"
    assert "resolver_node" in trace, f"resolver_node should be in trace: {trace}"
    assert trace[-1] == "supervisor_final_node", f"Last node should be supervisor_final_node, got: {trace[-1]}"


def test_e2e_tool_results_recorded():
    result = _process_ticket("How do I reserve an event?")
    tool_results = result.get("tool_results", [])
    kb_results = [t for t in tool_results if t.get("tool") == "kb_search"]
    assert len(kb_results) > 0, "kb_search should be recorded in tool_results"
    assert kb_results[0].get("matches_found", 0) > 0, "kb_search should find matches"


def test_e2e_support_tools_invoked():
    result = _process_ticket("I want a refund for my last reservation", user_email="refund@test.com")
    tool_results = result.get("tool_results", [])
    support_tools = [t for t in tool_results if t.get("tool") in ("account_lookup", "process_refund")]
    assert len(support_tools) > 0, f"Support tools should be invoked, got: {tool_results}"


def test_e2e_structured_logs():
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        result = _process_ticket("How do I reserve?", ticket_id="log-test")
    finally:
        sys.stdout = old_stdout

    log_output = captured.getvalue()
    log_lines = [line for line in log_output.strip().split("\n") if line.strip()]

    parsed_logs = []
    for line in log_lines:
        try:
            parsed_logs.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    agents_found = [log.get("agent") for log in parsed_logs]
    assert "supervisor_node" in agents_found, "supervisor_node log missing"
    assert "classifier_node" in agents_found, "classifier_node log missing"
    assert "resolver_node" in agents_found, "resolver_node log missing"
    assert "supervisor_final_node" in agents_found, "supervisor_final_node log missing"

    for log in parsed_logs:
        assert "timestamp" in log, f"Missing timestamp in log: {log}"
        assert "agent" in log, f"Missing agent in log: {log}"
        assert "action" in log, f"Missing action in log: {log}"
        assert "details" in log, f"Missing details in log: {log}"


if __name__ == "__main__":
    test_e2e_resolved_ticket()
    test_e2e_escalated_ticket()
    test_e2e_state_shape()
    test_e2e_billing_ticket()
    test_e2e_agent_trace_strict_order()
    test_e2e_tool_results_recorded()
    test_e2e_support_tools_invoked()
    test_e2e_structured_logs()
    print("All workflow tests passed!")
