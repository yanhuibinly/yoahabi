from pathlib import Path
from pprint import pprint
from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .decorators import log_io
import pandas as pd
import logging
import asyncio
import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult


logger = logging.getLogger(__name__)

PROJECT_DIRECTORY = "/Users/yanhuibin/myData/graphRAG"
graphrag_config = load_config(Path(PROJECT_DIRECTORY))


# index
# index_result: list[PipelineRunResult] =  await api.build_index(config=graphrag_config)

# for workflow_result in index_result:
#     status = f"error\n{workflow_result.errors}" if workflow_result.errors else "success"
#     print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")



async def _async_graph_search(query: str) -> str:
    """异步执行graph search的内部函数"""
    entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
    communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
    community_reports = pd.read_parquet(
        f"{PROJECT_DIRECTORY}/output/community_reports.parquet"
    )
    response, context = await api.global_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
    )
    return response


@tool
@log_io
def graph_retriever(
    query: Annotated[str, "The query to search from graph rag."],
) -> HumanMessage:
    """Use this to search from graph rag."""
    try:
        # 使用asyncio.run在同步上下文中运行异步函数
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果已经有事件循环在运行，需要使用不同的方法
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_graph_search(query))
                response = future.result()
        except RuntimeError:
            # 没有运行的事件循环，可以直接使用asyncio.run
            response = asyncio.run(_async_graph_search(query))
        
        return HumanMessage(content=response)
    except BaseException as e:
        error_msg = f"Failed to query graph rag. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
    