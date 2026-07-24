"""Direct tool loaders - bypass MCP subprocess for reliability.

These functions load tool functions directly without MCP server overhead.
Used as fallback when MCP tool loading fails (e.g., in Jupyter notebooks).
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))


def get_kb_search_tool():
    """Return kb_search as a callable."""
    from agentic.tools.kb_search_tool import kb_search
    return kb_search


def get_account_lookup_tool():
    """Return account_lookup as a callable."""
    from agentic.tools.account_lookup_tool import account_lookup
    return account_lookup


def get_refund_tool():
    """Return process_refund as a callable."""
    from agentic.tools.refund_tool import process_refund
    return process_refund


def get_read_memory_tool():
    """Return read_memory as a callable."""
    from agentic.tools.memory_tool import read_memory
    return read_memory


def get_write_memory_tool():
    """Return write_memory as a callable."""
    from agentic.tools.memory_tool import write_memory
    return write_memory
