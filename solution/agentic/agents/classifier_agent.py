"""Classifier Agent - reads ticket text + metadata, assigns category, urgency, and routing label.

Role:
    Analyze incoming ticket text and metadata to classify into one of six categories,
    determine urgency, and suggest a routing label.

Node function:
    classifier_node(state) -> dict  (partial state update)

Input (from state):
    - ticket_text: str
    - user_email: str
    - memory_context: list[dict]

Output (on state):
    - classification: dict {category, urgency, routing_label}
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import log_agent_step


class ClassificationOutput(BaseModel):
    category: str = Field(description="One of: billing, account, technical, subscription, content, onboarding")
    urgency: str = Field(description="One of: low, medium, high")
    routing_label: str = Field(description="One of: resolver, escalation")


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support ticket classifier for CultPass, a cultural experiences subscription platform. "
        "Classify the following customer ticket into one category, urgency level, and routing decision.\n\n"
        "Categories:\n"
        "- billing: charges, refunds, payment issues, double charges\n"
        "- account: login, password, profile, email changes, account deletion\n"
        "- technical: app crashes, QR code issues, streaming, bugs\n"
        "- subscription: plan changes, upgrades, downgrades\n"
        "- content: experience ratings, reviews, platform features\n"
        "- onboarding: getting started, app setup, first use\n\n"
        "Urgency:\n"
        "- high: financial loss, account lockout, security, urgent service interruption\n"
        "- medium: subscription changes, technical issues with workaround\n"
        "- low: how-to questions, feature inquiries, general info\n\n"
        "Routing:\n"
        "- resolver: ticket can likely be answered from knowledge base\n"
        "- escalation: requires human intervention (financial disputes, security, account deletion)\n\n"
        "Return a JSON object with fields: category, urgency, routing_label."
    ),
    ("human", "Ticket: {ticket_text}\nUser email: {user_email}"),
])


def classifier_node(state: dict) -> dict:
    """Classify the incoming ticket and return a partial state update with classification."""
    ticket_text = state.get("ticket_text", "")
    user_email = state.get("user_email", "")

    trace = state.get("agent_trace", [])
    result_trace = trace + ["classifier_node"]

    if not ticket_text:
        return {
            "classification": {
                "category": "unknown",
                "urgency": "low",
                "routing_label": "escalation",
            },
            "agent_trace": result_trace,
        }

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ClassificationOutput)
        chain = CLASSIFIER_PROMPT | structured_llm
        result = chain.invoke({"ticket_text": ticket_text, "user_email": user_email})
        classification = {
            "category": result.category,
            "urgency": result.urgency,
            "routing_label": result.routing_label,
        }
        log_agent_step(
            agent="classifier_node",
            action="classify_ticket",
            details=classification,
            ticket_id=state.get("ticket_id", ""),
        )
        return {
            "classification": classification,
            "agent_trace": result_trace,
        }
    except Exception:
        fallback = {
            "category": "unknown",
            "urgency": "low",
            "routing_label": "escalation",
        }
        log_agent_step(
            agent="classifier_node",
            action="classify_ticket_fallback",
            details=fallback,
            ticket_id=state.get("ticket_id", ""),
        )
        return {
            "classification": fallback,
            "agent_trace": result_trace,
        }
