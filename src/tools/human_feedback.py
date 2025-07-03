import logging
from langchain_core.tools import tool
from .decorators import log_io
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


@tool
@log_io
def human_feedback_tool(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
