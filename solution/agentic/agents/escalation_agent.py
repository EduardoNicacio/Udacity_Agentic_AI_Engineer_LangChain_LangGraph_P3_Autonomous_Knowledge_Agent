"""Escalation Agent - generates structured escalation summary for human handoff.

Role:
    When the resolver's confidence is below 0.6, this agent creates a detailed
    escalation summary including reason, priority, customer context, and suggested actions.

Node function:
    escalation_node(state) -> dict  (partial state update)

Input (from state):
    - ticket_text: str
    - classification: dict
    - resolution: dict
    - memory_context: list[dict]
    - tool_results: list[dict]

Output (on state):
    - escalation: dict {reason, priority, customer_context, suggested_actions}
    - tool_results: list[dict] (appended)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


ESCALATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an escalation specialist for CultPass customer support. "
        "Generate a structured escalation summary for a ticket that could not be resolved automatically.\n\n"
        "Return a JSON object with these fields:\n"
        '- "reason": string explaining why escalation is needed\n'
        '- "priority": one of "low", "medium", "high"\n'
        '- "customer_context": string with relevant customer info\n'
        '- "suggested_actions": array of strings with recommended next steps'
    ),
    ("human", "Ticket: {ticket_text}\nCategory: {category}\nUrgency: {urgency}\n"
              "Resolution attempt confidence: {confidence}\n"
              "Resolution response: {resolution_response}\n"
              "Customer history: {customer_history}"),
])


def escalation_node(state: dict) -> dict:
    """Generate an escalation summary and return a partial state update."""
    ticket_text = state.get("ticket_text", "")
    classification = state.get("classification", {})
    resolution = state.get("resolution", {})
    memory = state.get("memory_context", [])

    category = classification.get("category", "unknown") if classification else "unknown"
    urgency = classification.get("urgency", "low") if classification else "low"
    confidence = resolution.get("confidence", 0.0) if resolution else 0.0
    resolution_response = resolution.get("response", "") if resolution else ""

    customer_history = ""
    if memory:
        summaries = []
        for m in memory[:3]:
            s = m.get("summary", "")
            rt = m.get("resolution_type", "unknown")
            if s:
                summaries.append(f"[{rt}] {s}")
        customer_history = "; ".join(summaries)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = ESCALATION_PROMPT | llm
        result = chain.invoke({
            "ticket_text": ticket_text,
            "category": category,
            "urgency": urgency,
            "confidence": confidence,
            "resolution_response": resolution_response or "No resolution attempted",
            "customer_history": customer_history or "No prior history",
        })

        import json
        try:
            parsed = json.loads(result.content.strip().strip("`").replace("json\n", ""))
        except (json.JSONDecodeError, ValueError):
            parsed = {
                "reason": "Could not parse escalation summary",
                "priority": urgency,
                "customer_context": customer_history or "Unknown",
                "suggested_actions": ["Review ticket manually"],
            }
    except Exception:
        parsed = {
            "reason": "Error generating escalation summary",
            "priority": urgency,
            "customer_context": customer_history or "Unknown",
            "suggested_actions": ["Review ticket manually"],
        }

    tool_results = state.get("tool_results", [])
    tool_results.append({
        "tool": "escalation",
        "priority": parsed.get("priority", urgency),
        "reason": parsed.get("reason", ""),
    })

    trace = state.get("agent_trace", [])
    result_trace = trace + ["escalation_node"]

    return {
        "escalation": {
            "reason": parsed.get("reason", ""),
            "priority": parsed.get("priority", urgency),
            "customer_context": parsed.get("customer_context", ""),
            "suggested_actions": parsed.get("suggested_actions", []),
        },
        "tool_results": tool_results,
        "agent_trace": result_trace,
    }
