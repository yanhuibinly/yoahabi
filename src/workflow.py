import logging
from .config import TEAM_MEMBER_CONFIGRATIONS, TEAM_MEMBERS
from .graph import build_graph
from langchain_mcp_adapters.client import MultiServerMCPClient

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


async def init_mcp_tools():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/yanhuibin/myData",
                    "/Users/yanhuibin/myData",
                ],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    return tools


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

    # 初始化 MCP 工具
    import asyncio
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_running_loop()
        # 如果已经有事件循环在运行，使用create_task
        MCP_TOOLS = []  # 暂时设为空，避免循环嵌套问题
    except RuntimeError:
        # 没有运行的事件循环，可以使用asyncio.run
        MCP_TOOLS = asyncio.run(init_mcp_tools())

    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            "TEAM_MEMBER_CONFIGRATIONS": TEAM_MEMBER_CONFIGRATIONS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": False,
            "search_before_planning": False,
            # "MCP_TOOLS": MCP_TOOLS,
        },
        config={"configurable": {"thread_id": "default_thread"}}
    )
    logger.debug(f"Final workflow state: {result}")
    logger.info("Workflow completed successfully")
    return result


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
