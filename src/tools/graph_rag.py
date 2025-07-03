from pathlib import Path
from pprint import pprint
from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .decorators import log_io
import pandas as pd
import logging
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



@tool
@log_io
async def graph_retrieve(
    query: Annotated[str, "The query to search from graph rag."],
) -> HumanMessage:
    """Use this to search from graph rag."""
    try:
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
        return HumanMessage(content=response)
    except BaseException as e:
        error_msg = f"Failed to query graph rag. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
    