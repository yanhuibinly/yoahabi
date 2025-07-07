import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# TWITTER_TOOLS = []

async def init_twitter_tools():
    """Initialize MCP tools asynchronously."""
    try:
        client = MultiServerMCPClient(
            {
                "x-twitter-mcp": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        "/Users/yanhuibin/Documents/aicode/yoahabi",
                        "run",
                        "x-twitter-mcp-server"
                    ],
                    "env": {
                        "PYTHONUNBUFFERED": "1"
                    },
                    "transport": "stdio"
                }
            }
        )
        tools = await client.get_tools()
        return tools;
    except Exception as e:
        logger.warning(f"Failed to initialize MCP tools: {e}")
        return []


def get_twitter_tools():
    """Get MCP tools, initializing them if needed."""
    try:
        # 使用asyncio.run在同步上下文中运行异步函数
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果已经有事件循环在运行，需要使用不同的方法
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, init_twitter_tools())
                response = future.result()
        except RuntimeError:
            # 没有运行的事件循环，可以直接使用asyncio.run
            response = asyncio.run(init_twitter_tools())
        logger.info(f"Initialized {len(response)} MCP tools")
        return response
    except BaseException as e:
        logger.warning(f"Failed to initialize MCP tools: {e}")
        return []
    

if __name__ == "__main__":
    # 测试 MCP 工具初始化
    tools = get_twitter_tools()
    print(f"MCP Tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}")