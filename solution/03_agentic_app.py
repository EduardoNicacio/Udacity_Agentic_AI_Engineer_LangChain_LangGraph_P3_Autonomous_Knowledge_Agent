"""
UDA-Hub - LangGraph-powered multi-agent customer support automation for CultPass.

Requires Python 3.11+.

Run instructions:
    cd solution/
    pip install -r requirements.txt
    python -m ipykernel install --user --name=udahub  (if running notebooks)
    jupyter notebook 01_external_db_setup.ipynb          # Set up CultPass external DB
    jupyter notebook 02_core_db_setup.ipynb              # Set up UDA-Hub core DB
    python 03_agentic_app.py                             # Launch interactive chat loop

The chat loop processes one ticket per session. Each run begins with a prompt
for the customer's issue. The agent graph classifies, resolves (via KB RAG),
and either provides an answer or escalates to human support.
"""

from dotenv import load_dotenv
from agentic.workflow import compile_graph
from utils import chat_interface


def main():
    load_dotenv()

    graph = compile_graph()

    ticket_id = input("Enter ticket ID (or press Enter for default 'ticket-1'): ").strip()
    if not ticket_id:
        ticket_id = "ticket-1"

    chat_interface(graph, ticket_id)


if __name__ == "__main__":
    main()
