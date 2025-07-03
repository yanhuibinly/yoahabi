import asyncio
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type, Any, Dict, List
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.llms.llm import vl_llm
from src.tools.decorators import create_logged_tool
from src.config import (
    CHROME_INSTANCE_PATH,
    CHROME_HEADLESS,
    CHROME_PROXY_SERVER,
    CHROME_PROXY_USERNAME,
    CHROME_PROXY_PASSWORD,
    BROWSER_HISTORY_DIR,
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class LLMWrapper:
    """Wrapper for LLM objects to make them compatible with browser_use library."""
    
    def __init__(self, llm):
        self.llm = llm
        # Copy all attributes from the original LLM
        for attr in dir(llm):
            if not attr.startswith('_') and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(llm, attr))
                except:
                    pass  # Skip attributes that can't be copied
        
        # Add browser_use specific attributes if missing
        if not hasattr(self, 'provider'):
            self.provider = 'openai'  # Default provider
        if not hasattr(self, 'model'):
            self.model = getattr(llm, 'model_name', getattr(llm, 'model', 'gpt-4'))  # Fallback model name
        
        # Additional browser_use compatibility attributes
        if not hasattr(self, 'model_name'):
            self.model_name = self.model
        if not hasattr(self, 'temperature'):
            self.temperature = getattr(llm, 'temperature', 0.0)
    
    def _convert_browser_use_messages(self, messages):
        """Convert browser_use message types to LangChain message types."""
        converted_messages = []
        for msg in messages:
            # Check if it's a browser_use message type
            if hasattr(msg, '__class__') and 'browser_use' in str(msg.__class__):
                msg_type = msg.__class__.__name__
                content = getattr(msg, 'content', '')
                
                # Extract additional attributes for compatibility
                additional_kwargs = {}
                if hasattr(msg, 'name'):
                    additional_kwargs['name'] = msg.name
                if hasattr(msg, 'additional_kwargs'):
                    additional_kwargs.update(msg.additional_kwargs)
                
                logger.debug(f"Converting browser_use {msg_type} to LangChain message")
                
                if msg_type == 'SystemMessage':
                    converted_messages.append(SystemMessage(content=content, **additional_kwargs))
                elif msg_type == 'HumanMessage' or msg_type == 'UserMessage':
                    converted_messages.append(HumanMessage(content=content, **additional_kwargs))
                elif msg_type == 'AIMessage' or msg_type == 'AssistantMessage':
                    # Add usage attribute if missing for browser_use compatibility
                    ai_msg = AIMessage(content=content, **additional_kwargs)
                    if not hasattr(ai_msg, 'usage') and hasattr(msg, 'usage'):
                        ai_msg.usage = msg.usage
                    elif not hasattr(ai_msg, 'usage'):
                        # Add empty usage for compatibility
                        ai_msg.usage = {}
                    converted_messages.append(ai_msg)
                else:
                    # Fallback to HumanMessage for unknown types
                    logger.warning(f"Unknown browser_use message type: {msg_type}, converting to HumanMessage")
                    converted_messages.append(HumanMessage(content=str(content), **additional_kwargs))
            else:
                # Already a LangChain message or compatible format
                converted_messages.append(msg)
        return converted_messages
    
    def _ensure_browser_use_compatibility(self, result):
        """Ensure the result has all attributes browser_use expects."""
        if hasattr(result, 'content'):
            # Add usage attribute
            if not hasattr(result, 'usage'):
                usage_data = getattr(result, 'usage_metadata', {})
                if not usage_data and hasattr(result, 'response_metadata'):
                    usage_data = result.response_metadata.get('token_usage', {})
                if not usage_data:
                    usage_data = {}
                
                try:
                    result.usage = usage_data
                except AttributeError:
                    setattr(result, 'usage', usage_data)
            
            # Add completion attribute for browser_use
            if not hasattr(result, 'completion'):
                try:
                    result.completion = result.content
                except AttributeError:
                    setattr(result, 'completion', result.content)
            
            # Convert content format if it's a list (multimodal)
            if isinstance(result.content, list):
                # Convert multimodal format to simple string
                content_str = ""
                for part in result.content:
                    # Handle ContentPartTextParam objects
                    if hasattr(part, 'text'):
                        content_str += str(part.text)
                    # Handle dict format
                    elif isinstance(part, dict) and 'text' in part:
                        content_str += str(part['text'])
                    # Handle string parts
                    elif isinstance(part, str):
                        content_str += part
                    # Handle other text-like attributes
                    elif hasattr(part, '__str__') and 'text' in str(part).lower():
                        # Try to extract text from the string representation
                        part_str = str(part)
                        if 'text=' in part_str:
                            try:
                                # Extract text content from string representation
                                text_start = part_str.find('text=') + 5
                                if part_str[text_start] in ['"', "'"]:
                                    quote = part_str[text_start]
                                    text_start += 1
                                    text_end = part_str.find(quote, text_start)
                                    if text_end != -1:
                                        content_str += part_str[text_start:text_end]
                            except:
                                pass
                    # Skip image parts and unknown formats
                
                # Only update if we extracted some text, otherwise use fallback
                if content_str.strip():
                    try:
                        result.content = content_str.strip()
                    except AttributeError:
                        setattr(result, 'content', content_str.strip())
                        
                    # Update completion as well
                    try:
                        result.completion = content_str.strip()
                    except AttributeError:
                        setattr(result, 'completion', content_str.strip())
                else:
                    # Fallback: convert the entire list to string
                    fallback_content = str(result.content)
                    try:
                        result.content = fallback_content
                        result.completion = fallback_content
                    except AttributeError:
                        setattr(result, 'content', fallback_content)
                        setattr(result, 'completion', fallback_content)
        
        return result
    
    async def ainvoke(self, *args, **kwargs) -> Any:
        """Async invoke method that browser_use expects."""
        # Extract messages from args or kwargs
        if args:
            messages = args[0]
            # Convert browser_use messages to LangChain messages
            if isinstance(messages, list):
                messages = self._convert_browser_use_messages(messages)
            # Pass remaining args and kwargs to the underlying LLM
            result = await self.llm.ainvoke(messages, **kwargs)
        elif 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, list):
                messages = self._convert_browser_use_messages(messages)
            result = await self.llm.ainvoke(messages, **{k: v for k, v in kwargs.items() if k != 'messages'})
        else:
            result = await self.llm.ainvoke(*args, **kwargs)
        
        # Ensure browser_use compatibility
        result = self._ensure_browser_use_compatibility(result)
        
        return result
    
    def invoke(self, *args, **kwargs) -> Any:
        """Sync invoke method."""
        # Extract messages from args or kwargs
        if args:
            messages = args[0]
            # Convert browser_use messages to LangChain messages
            if isinstance(messages, list):
                messages = self._convert_browser_use_messages(messages)
            result = self.llm.invoke(messages, **kwargs)
        elif 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, list):
                messages = self._convert_browser_use_messages(messages)
            result = self.llm.invoke(messages, **{k: v for k, v in kwargs.items() if k != 'messages'})
        else:
            result = self.llm.invoke(*args, **kwargs)
        
        # Ensure browser_use compatibility
        result = self._ensure_browser_use_compatibility(result)
        
        return result
    
    def __getattr__(self, name):
        """Fallback to the wrapped LLM for any missing attributes."""
        # Handle specific browser_use attributes
        if name == 'provider':
            return 'openai'
        if name == 'model' and hasattr(self.llm, 'model_name'):
            return self.llm.model_name
        if name == 'model' and hasattr(self.llm, 'model'):
            return self.llm.model
        return getattr(self.llm, name)
    
    def __setattr__(self, name, value):
        """Allow setting attributes that browser_use might need."""
        if name in ['llm'] or name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # Allow setting additional attributes for browser_use compatibility
            super().__setattr__(name, value)


browser_config = BrowserConfig(
    headless=CHROME_HEADLESS,
    chrome_instance_path=CHROME_INSTANCE_PATH,
)
if CHROME_PROXY_SERVER:
    proxy_config = {
        "server": CHROME_PROXY_SERVER,
    }
    if CHROME_PROXY_USERNAME:
        proxy_config["username"] = CHROME_PROXY_USERNAME
    if CHROME_PROXY_PASSWORD:
        proxy_config["password"] = CHROME_PROXY_PASSWORD
    browser_config.proxy = proxy_config

expected_browser = Browser(config=browser_config)


class BrowserUseInput(BaseModel):
    """Input for WriteFileTool."""

    instruction: str = Field(..., description="The instruction to use browser")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "Use this tool to interact with web browsers. Input should be a natural language description of what you want to do with the browser, such as 'Go to google.com and search for browser-use', or 'Navigate to Reddit and find the top post about AI'."
    )

    _agent: Optional[BrowserAgent] = None

    def _generate_browser_result(
        self, result_content: str, generated_gif_path: str
    ) -> dict:
        return {
            "result_content": result_content,
            "generated_gif_path": generated_gif_path,
        }

    def _run(self, instruction: str) -> str:
        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        """Run the browser task synchronously."""
        # Wrap the LLM to make it compatible with browser_use
        wrapped_llm = LLMWrapper(vl_llm)
        
        self._agent = BrowserAgent(
            task=instruction,  # Will be set per request
            llm=wrapped_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._agent.run())

                if isinstance(result, AgentHistoryList):
                    return json.dumps(
                        self._generate_browser_result(
                            result.final_result(), generated_gif_path
                        )
                    )
                else:
                    return json.dumps(
                        self._generate_browser_result(result, generated_gif_path)
                    )
            finally:
                loop.close()
        except Exception as e:
            return f"Error executing browser task: {str(e)}"

    async def terminate(self):
        """Terminate the browser agent if it exists."""
        if self._agent and self._agent.browser:
            try:
                await self._agent.browser.close()
            except Exception as e:
                logger.error(f"Error terminating browser agent: {str(e)}")
        self._agent = None

    async def _arun(self, instruction: str) -> str:
        """Run the browser task asynchronously."""
        generated_gif_path = f"{BROWSER_HISTORY_DIR}/{uuid.uuid4()}.gif"
        # Wrap the LLM to make it compatible with browser_use
        wrapped_llm = LLMWrapper(vl_llm)
        
        self._agent = BrowserAgent(
            task=instruction,
            llm=wrapped_llm,
            browser=expected_browser,
            generate_gif=generated_gif_path,  # Will be set per request
        )
        try:
            result = await self._agent.run()
            if isinstance(result, AgentHistoryList):
                return json.dumps(
                    self._generate_browser_result(
                        result.final_result(), generated_gif_path
                    )
                )
            else:
                return json.dumps(
                    self._generate_browser_result(result, generated_gif_path)
                )
        except Exception as e:
            return f"Error executing browser task: {str(e)}"
        finally:
            await self.terminate()


BrowserTool = create_logged_tool(BrowserTool)
browser_tool = BrowserTool()

if __name__ == "__main__":
    browser_tool._run(instruction="go to search anything")
