from .env import (
    # Reasoning LLM
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision-language LLM
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    # Other configurations
    CHROME_INSTANCE_PATH,
    CHROME_HEADLESS,
    CHROME_PROXY_SERVER,
    CHROME_PROXY_USERNAME,
    CHROME_PROXY_PASSWORD,
)
from .tools import TAVILY_MAX_RESULTS, BROWSER_HISTORY_DIR

# Team configuration
TEAM_MEMBER_CONFIGRATIONS = {
    "researcher": {
        "name": "researcher",
        "desc": (
            "Responsible for searching and collecting relevant information, understanding user needs and conducting research analysis"
        ),
        "desc_for_llm": (
            "Uses search engines and web crawlers to gather information from the internet. "
            "Outputs a Markdown report summarizing findings. Researcher can not do math or programming."
        ),
        "is_optional": False,
    },
    "coder": {
        "name": "coder",
        "desc": (
            "Responsible for code implementation, debugging and optimization, handling technical programming tasks"
        ),
        "desc_for_llm": (
            "Executes Python or Bash commands, performs mathematical calculations, and outputs a Markdown report. "
            "Must be used for all mathematical computations."
        ),
        "is_optional": True,
    },
    "browser": {
        "name": "browser",
        "desc": "Responsible for web browsing, content extraction and interaction",
        "desc_for_llm": (
            "Directly interacts with web pages, performing complex operations and interactions. "
            "You can also leverage `browser` to perform in-domain search, like Facebook, Instgram, Github, etc."
        ),
        "is_optional": True,
    },
    "reporter": {
        "name": "reporter",
        "desc": (
            "Responsible for summarizing analysis results, generating reports and presenting final outcomes to users"
        ),
        "desc_for_llm": "Write a professional report based on the result of each step.",
        "is_optional": False,
    },
    "rag": {
        "name": "rag",
        "desc": (
             "Responsible for retrieving information using RAG (Retrieval-Augmented Generation) technology"
         ),
         "desc_for_llm": (
             "Uses RAG to query and retrieve relevant information from various knowledge sources including documents, knowledge graphs, and vector databases. "
             "Provides comprehensive answers based on retrieved context and semantic search capabilities."
         ),
        "is_optional": True,
    },
}

TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGRATIONS.keys())

__all__ = [
    # Reasoning LLM
    "REASONING_MODEL",
    "REASONING_BASE_URL",
    "REASONING_API_KEY",
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision-language LLM
    "VL_MODEL",
    "VL_BASE_URL",
    "VL_API_KEY",
    # Other configurations
    "TEAM_MEMBERS",
    "TEAM_MEMBER_CONFIGRATIONS",
    "TAVILY_MAX_RESULTS",
    "CHROME_INSTANCE_PATH",
    "CHROME_HEADLESS",
    "CHROME_PROXY_SERVER",
    "CHROME_PROXY_USERNAME",
    "CHROME_PROXY_PASSWORD",
    "BROWSER_HISTORY_DIR",
]
