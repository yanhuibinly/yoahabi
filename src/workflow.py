import logging
from .config import TEAM_MEMBER_CONFIGRATIONS, TEAM_MEMBERS
from .graph import build_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()


def run_agent_workflow(user_input: str, debug: bool = False):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input}")

    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            "TEAM_MEMBER_CONFIGRATIONS": TEAM_MEMBER_CONFIGRATIONS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": False,
            "search_before_planning": False,
        },
        config={"configurable": {"thread_id": "default_thread"}}
    )
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
