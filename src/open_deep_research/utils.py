"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import warnings
from urllib.parse import urljoin, urlsplit
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
import httpx
from bs4 import BeautifulSoup
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research import observability as obs
from open_deep_research.artifacts import ArtifactStore, Identity, ToolResult
from open_deep_research.context import clip_preserving_spans
from open_deep_research.configuration import Configuration, ContextPolicy, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.recovery import request_context, request_context_text
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # Step 3: resolve ONE call context for the run (config, model, budget, execution).
    from open_deep_research.context import call_context_for

    call_context = call_context_for(config, role="summarization")
    configurable = call_context.config

    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length

    # Raw chat model: ``assemble`` binds the response schema and owns the retry, so a
    # retry is one reserved attempt, not an inner ``with_retry`` plus an outer one.
    # ``api_key=None`` must not be passed explicitly: it overrides the provider SDK's
    # own environment lookup (ChatDeepSeek rejects a None key outright).
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    model_kwargs: Dict[str, Any] = {
        "model": configurable.summarization_model,
        "max_tokens": configurable.summarization_model_max_tokens,
        "tags": ["langsmith:nostream"],
    }
    if model_api_key:
        model_kwargs["api_key"] = model_api_key
    options = provider_request_options(configurable.summarization_model)
    if options:
        model_kwargs["extra_body"] = options
    summarization_model = init_chat_model(**model_kwargs)
    
    # Step 4: Create summarization tasks (skip empty content)
    async def noop():
        """No-op function for results without raw content."""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include],
            call_context=call_context,
        )
        for result in unique_results.values()
    ]
    
    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)
    
    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    from open_deep_research.context import call_context_for

    search_context = call_context_for(config or {}, role="search")
    search_store = search_context.store
    search_request_id = ""
    maximum = search_context.config.max_total_search_queries
    if search_store is not None and maximum is not None:
        search_request_id = search_store.reserve_search_queries(len(search_queries), maximum, execution_id=search_context.execution_id)
    # Initialize the Tavily client only after the query budget admits the batch.
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
            **({"include_domains": search_context.config.search_domains} if search_context.config.search_domains else {}),
        )
        for query in search_queries
    ]
    
    # Execute all search queries in parallel and return results
    search_span = obs.span("search.raw", kind=obs.SPAN_STAGE, attributes={"query_count": len(search_tasks)})
    try:
        with search_span:
            search_results = await asyncio.gather(*search_tasks)
    except BaseException:
        if search_request_id:
            search_store.append_ledger("search_usage", {"request_id": search_request_id, "phase": "finished", "status": "failed_or_cancelled",
                                                        "provider_credits": None, "provider_billed_cost_usd": None})
        raise
    if search_request_id:
        # Preserve provider usage when returned; absence is unknown, never zero.
        search_store.append_ledger("search_usage", {"request_id": search_request_id, "phase": "finished", "status": "returned",
                                                    "provider_usage": [row.get("usage") for row in search_results],
                                                    "provider_credits": None, "provider_billed_cost_usd": None})
    return search_results

async def summarize_webpage(
    model: BaseChatModel,
    webpage_content: str,
    *,
    call_context: Any = None,
    config: RunnableConfig = None,
    store: Optional[ArtifactStore] = None,
    identity: Optional[Identity] = None,
) -> str:
    """Summarize webpage content through the run ledger, with timeout protection.

    The original one-message prompt and ``Summary`` schema are preserved verbatim: the
    call is wrapped for accounting, not rewritten by the scoped context builder. The
    caller passes the *unwrapped* chat model so ``assemble`` owns the single
    application-level retry (an inner ``with_retry`` would double-count attempts).

    Args:
        model: The chat model configured for summarization (no retry wrapper)
        webpage_content: Raw webpage content to be summarized
        call_context: The run's resolved configuration/budget/execution (preferred)
        config: Fallback runnable config used to resolve a call context

    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    from open_deep_research.context import assemble, auxiliary_packet, call_context_for

    context = call_context or (call_context_for(config, role="summarization", store=store) if config is not None else None)
    extract_span = obs.span("page.extract", kind=obs.SPAN_STAGE, attributes={"channel": "baseline_summarize", "role": "summarization"})
    try:
        # Create prompt with current date context (unchanged baseline prompt)
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )
        messages = [HumanMessage(content=prompt_content)]
        with extract_span:
            if context is not None:
                packet = auxiliary_packet(
                    context,
                    messages,
                    response_schema=Summary,
                    max_output_tokens=context.config.summarization_model_max_tokens,
                    identity=identity,
                )
                summary = await asyncio.wait_for(
                    assemble(
                        model,
                        packet,
                        store=context.store if context.config.artifacts_enabled else None,
                        record_requests=context.config.record_requests,
                        max_tokens=context.config.summarization_model_max_tokens,
                        model_name=context.config.summarization_model,
                        retries=context.config.max_structured_output_retries,
                        budget=context.budget,
                    ),
                    timeout=60.0,
                )
            else:
                # No run context at all (direct library use): the call cannot be metered.
                # Say so instead of reporting a smaller total as if it were complete.
                summary = await asyncio.wait_for(model.ainvoke(messages), timeout=60.0)
                obs.record_event("auxiliary_usage_unmetered", role="summarization", model=getattr(model, "model_name", "") or "", reason_code="no_run_context")
        if context is not None and context.store is None:
            obs.record_event("auxiliary_usage_unmetered", role="summarization", model=context.config.summarization_model, reason_code="no_ledger")

        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        extract_span.fail(error_class="TimeoutError", status=obs.STATUS_TIMEOUT)
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        extract_span.fail(error_class=type(e).__name__)
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []
        
    # Default fallback for unknown search API types
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # Context recovery is bound only in the scoped policy, where the assembler actually
    # trims material and points at this tool. Binding it in the baseline path would
    # change the A/C arm under the name of a no-op.
    if configurable.artifacts_enabled and configurable.context_policy == ContextPolicy.SCOPED:
        tools.append(request_context)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Tool registry / capability mapping
##########################
# Capability tags select tools by what the task needs, not by keyword similarity.
# Tools exposing the same capability (a native web-search spec, tavily_search or any
# search adapter named web_search) are equivalent for selection purposes.
CAPABILITY_MAP: Dict[str, Dict[str, str]] = {
    "research_dispatch": {"ConductResearch": "research_dispatch"},
    "research_done": {"ResearchComplete": "research_done", "ResearchCompleted": "research_done"},
    "self_reflection": {"think_tool": "self_reflection", "ThinkTool": "self_reflection"},
    "context_recovery": {"request_context": "context_recovery"},
    "web_search": {"web_search": "web_search", "tavily_search": "web_search"},
    "repo_read": {
        "read_file": "repo_read",
        "list_files": "repo_read",
        "grep": "repo_read",
        "search_code": "repo_read",
        "github": "repo_read",
    },
    "paper_search": {"arxiv_search": "paper_search", "semantic_scholar": "paper_search"},
}

TOOL_CAPABILITIES: Dict[str, set] = {}
for _tag, _names in CAPABILITY_MAP.items():
    for _name in _names:
        TOOL_CAPABILITIES.setdefault(_name, set()).add(_tag)


@dataclass
class ToolSet:
    """One version of the tool selection bound to one model request."""

    version: str
    definitions: List[Any] = field(default_factory=list)
    executors: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """Return the tool names in this set."""
        return list(self.executors)


def tool_name(tool: Any) -> str:
    """Return the name of a tool object, dict or provider-native tool spec."""
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("type") or "")
    return str(getattr(tool, "name", "") or "")


def is_callable_tool(tool: Any) -> bool:
    """Return True when a tool has a Python executor (native search specs do not)."""
    return isinstance(tool, BaseTool) or callable(getattr(tool, "ainvoke", None))


def register_capabilities(name: str, tags: List[str]) -> None:
    """Bind extra capability tags to a tool name (MCP tools, plugins)."""
    for tag in tags:
        TOOL_CAPABILITIES.setdefault(name, set()).add(tag)


def capabilities_for(name: str) -> set:
    """Return capability tags for a tool name (unmapped names get their own tag)."""
    return TOOL_CAPABILITIES.get(name, {name})


# Words that indicate which *kind* of source a sub-task needs. The tag decides the
# tool intersection; it is a deterministic rule, not a semantic classifier, and the
# default always keeps ordinary web search (dropping it would silently cap research).
_SOURCE_HINTS: Dict[str, tuple] = {
    "repo_read": (
        "repo", "repository", "github", "gitlab", "source code", "codebase", "function",
        "implementation", "api doc", "sdk", "源码", "代码", "仓库", "函数", "实现", "接口",
    ),
    "paper_search": (
        "paper", "arxiv", "preprint", "journal", "literature", "citation", "study",
        "论文", "文献", "期刊", "预印本", "研究综述",
    ),
    "mcp_read": ("mcp", "model context protocol", "mcp server"),
}


def derive_task_capabilities(topic: str, *, artifacts_enabled: bool = True) -> List[str]:
    """Return the capability tags one sub-task needs, from its own description.

    Before this rule every task card carried the same five tags, so the intersection
    filter existed but never actually removed an unrelated source class. The rule is
    additive on top of web search (the default research source) and only narrows when
    the task text names a different source class; it never removes self-reflection or
    the archive read-back path.
    """
    lowered = (topic or "").lower()
    tags: List[str] = ["web_search", "self_reflection", "research_done"]
    for tag, hints in _SOURCE_HINTS.items():
        if any(hint in lowered for hint in hints):
            tags.append(tag)
    if artifacts_enabled:
        tags.append("context_recovery")
    return tags


_TASK_WORD_RE = re.compile(r"[A-Za-z0-9_./-]{3,}|[\u4e00-\u9fff]{2,}")


def match_task_requirements(topic: str, requirements: List[Any]) -> List[str]:
    """Return the requirement ids whose statement overlaps this sub-task's topic.

    When nothing matches, every requirement id is returned: refusing to guess is safer
    than silently dropping a user constraint from a sub-task.
    """
    all_ids = [getattr(req, "requirement_id", "") for req in requirements or [] if getattr(req, "requirement_id", "")]
    topic_terms = {match.lower() for match in _TASK_WORD_RE.findall(topic or "")}
    if not topic_terms or not all_ids:
        return all_ids
    matched: List[str] = []
    for req in requirements or []:
        statement = getattr(req, "statement", "") or ""
        terms = [match.lower() for match in _TASK_WORD_RE.findall(statement)][:12]
        if not terms:
            continue
        overlap = len(topic_terms & set(terms))
        if overlap >= 1 and (overlap >= 2 or max(len(t) for t in terms) >= 6):
            matched.append(getattr(req, "requirement_id", ""))
    return matched or all_ids


def select_tools(
    role: str,
    capability_tags: List[str],
    registry: List[Any],
    *,
    extra: Optional[List[str]] = None,
) -> ToolSet:
    """Select the tools for one role/phase from an explicit registry.

    The selection is a set intersection, not a similarity search:
    ``registry ∩ deployment ∩ role capabilities ∩ task capabilities``.

    Unmapped registry entries are rejected, so a stale or forged tool name never
    silently falls back to "all tools".
    """
    allowed_tags = set(capability_tags) | set(extra or [])
    definitions: List[Any] = []
    executors: Dict[str, Any] = {}
    rejected: List[str] = []
    for tool in registry:
        name = tool_name(tool)
        if not name:
            continue
        if not (capabilities_for(name) & allowed_tags):
            rejected.append(name)
            continue
        executors[name] = tool
        if is_callable_tool(tool):
            definitions.append(tool)
        # Provider-native tools (e.g. {"type": "web_search_preview"}) are bound but have
        # no local executor and are therefore not in the executor table.
    payload = json.dumps(
        {"role": role, "tags": sorted(allowed_tags), "tools": sorted((n, sorted(capabilities_for(n))) for n in executors)},
        sort_keys=True,
        default=str,
    )
    return ToolSet(
        version=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        definitions=definitions,
        executors=executors,
        capabilities=sorted(allowed_tags),
        rejected=rejected,
    )


class ToolAccessDenied(Exception):
    """Raised when a model asks for a tool that is not bound in this request."""

    status = "tool_not_bound"


def classify_tool_error(exc: BaseException) -> str:
    """Map an exception to a stable, structured error class."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, asyncio.TimeoutError) or "timeout" in name or "timeout" in text:
        return "timeout"
    if "ratelimit" in name or "rate_limit" in text or "429" in text:
        return "rate_limited"
    if "auth" in name or "401" in text or "403" in text or "unauthorized" in text:
        return "auth_error"
    if "connection" in name or "network" in text:
        return "network_error"
    if isinstance(exc, ToolException):
        return "tool_error"
    return "unknown"


async def execute_tool_call(
    tool_call: Dict[str, Any],
    toolset: ToolSet,
    config: RunnableConfig,
) -> ToolResult:
    """Execute one call against a frozen tool set with a structured status.

    Unknown, expired or unbound names are rejected *before* execution; this path never
    re-resolves a full tool list.
    """
    name = tool_call.get("name") or ""
    args = dict(tool_call.get("args") or {})
    tool = toolset.executors.get(name)
    if tool is None or not is_callable_tool(tool):
        return ToolResult(
            status="error",
            error_class="tool_not_bound",
            error=f"Tool {name!r} is not bound in tool set {toolset.version}.",
            content=f"Error: tool {name!r} is not available in this request. Available tools: {', '.join(toolset.names) or '(none)'}.",
        )
    accepts_config = True
    try:
        params = list(inspect.signature(tool.ainvoke).parameters.values())
        accepts_config = any(p.kind is p.VAR_POSITIONAL for p in params) or any(p.name == "config" for p in params)
    except (TypeError, ValueError):
        accepts_config = True
    if accepts_config:
        args = {**args, "config": config}
    # The host binds the identity; ``__odr_identity__`` is never a model argument.
    identity = getattr(tool, "_odr_identity", None)
    if name == "request_context":
        # Read-back is executed by the host from the bound identity, so the model
        # cannot pass an agent id, a path or a permission flag.
        text = request_context_text(
            need=str(args.get("need") or ""),
            purpose=str(args.get("purpose") or ""),
            known_refs=args.get("known_refs") or [],
            expected_detail=str(args.get("expected_detail") or ""),
        )
        return ToolResult(status="ok", content=text)
    if identity is not None and tool_name(tool) == "tavily_search":
        result = await evidence_aware_web_search(
            list(args.get("queries") or []),
            identity=identity,
            store=getattr(tool, "_odr_store", None),
            config=config,
            max_results=int(args.get("max_results") or 5),
            topic=args.get("topic") or "general",
        )
        return result
    try:
        output = await tool.ainvoke(args)
        if isinstance(output, ToolResult):
            return output
        if isinstance(output, dict) and "status" in output and "content" in output:
            return ToolResult(**output)
        return ToolResult(status="ok", content=str(output))
    except Exception as exc:  # noqa: BLE001 - converted to a structured result
        error_class = classify_tool_error(exc)
        return ToolResult(
            status="error",
            error_class=error_class,
            error=f"{type(exc).__name__}: {exc}"[:500],
            content=(
                f"Tool error ({error_class}): {type(exc).__name__}: {str(exc)[:300]}. "
                "This is a tool failure, not evidence that the information does not exist."
            ),
        )


##########################
# Evidence-aware extraction
##########################

def official_domain_match(url: str, domain: str) -> bool:
    """Return True only when ``url`` is an https document on ``domain`` (or a subdomain).

    Substring checks accept ``postgresql.org.evil.test`` and
    ``evil.test/?u=postgresql.org``; neither is the official site. A missing scheme is
    also rejected: an archived URL must be a real document location.
    """
    from urllib.parse import urlparse

    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower().rstrip(".")
    wanted = str(domain or "").lower().strip(".").rstrip(".")
    if not parsed.scheme or parsed.scheme.lower() not in {"http", "https"} or not host or not wanted:
        return False
    return host == wanted or host.endswith("." + wanted)


def select_excerpt(content: str, *, query: str = "", limit: int = 400) -> str:
    """Pick a short, verbatim excerpt from ``content`` for evidence storage."""
    text = (content or "").strip()
    if not text:
        return ""
    start = 0
    if query:
        idx = text.lower().find(query.lower())
        if idx >= 0:
            start = max(0, idx - limit // 4)
    return text[start : start + limit]


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "is",
    "what", "when", "which", "how", "does", "did", "official", "please", "about",
    "into", "than", "then", "there", "their", "have", "has", "had", "not", "但", "的",
    "了", "和", "是", "在", "有", "与", "及", "或", "对", "为", "以",
}


def _relevance_terms(*texts: str) -> list[str]:
    """Return content-bearing terms for a task-relevant excerpt window."""
    terms: list[str] = []
    for text in texts:
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", str(text or "").lower()):
            token = token.strip(".-_")
            if len(token) < 3 and not re.match(r"[\u4e00-\u9fff]", token):
                continue
            if token in _STOPWORDS or token in terms:
                continue
            terms.append(token)
    return terms[:40]


def select_relevant_window(
    content: str,
    *,
    query: str = "",
    objective: str = "",
    limit: int = 1200,
) -> tuple[str, int, int]:
    """Return the window of ``content`` that best matches the task's own words.

    The old fixed head excerpt showed a page's navigation while the answer lived in
    the middle. Candidates are verbatim windows anchored on term occurrences; the
    highest-overlap window wins and ties fall back to the earliest one. Returns
    ``(text, start, end)`` so the caller can record exact offsets.
    """
    text = content or ""
    if len(text) <= limit:
        return text, 0, len(text)
    terms = _relevance_terms(query, objective)
    if not terms:
        return text[:limit], 0, limit
    lowered = text.lower()
    anchors: list[int] = []
    for term in terms[:20]:
        position = lowered.find(term)
        if position >= 0:
            anchors.append(position)
    if not anchors:
        return text[:limit], 0, limit
    best = (0, 0, -1)
    for anchor in anchors[:60]:
        start = max(0, anchor - limit // 4)
        newline = text.rfind("\n", max(0, start - 200), start + 200)
        if newline > 0:
            start = newline + 1
        end = min(len(text), start + limit)
        window = lowered[start:end]
        score = sum(1 for term in terms if term in window)
        if score > best[2]:
            best = (start, end, score)
    start, end, _ = best
    return text[start:end], start, end


def select_relevant_windows(
    content: str,
    *,
    query: str = "",
    objective: str = "",
    limit: int = 1200,
    max_windows: int = 2,
) -> list[tuple[int, int]]:
    """Return up to ``max_windows`` non-overlapping task-relevant ranges.

    A long page can hold two facts that the task needs (a GA date and a supported
    version, say). One window around the best anchor would hide the other; this picks
    the best distinct anchors instead. Ranges are verbatim and returned in document
    order so the caller can record offsets and omitted ranges.
    """
    text = content or ""
    if len(text) <= limit:
        return [(0, len(text))]
    terms = _relevance_terms(query, objective)
    if not terms:
        return [(0, min(limit, len(text)))]
    lowered = text.lower()
    candidates: list[tuple[int, int, int]] = []
    for term in terms[:20]:
        start_at = 0
        for _ in range(3):
            position = lowered.find(term, start_at)
            if position < 0:
                break
            start_at = position + len(term)
            start = max(0, position - limit // 3)
            newline = text.rfind("\n", max(0, start - 200), start + 200)
            if newline > 0:
                start = newline + 1
            end = min(len(text), start + limit)
            score = sum(1 for candidate_term in terms if candidate_term in lowered[start:end])
            candidates.append((score, start, end))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[int, int]] = []
    for score, start, end in candidates:
        if score < 1:
            break
        overlap = False
        for other_start, other_end in chosen:
            shared = min(end, other_end) - max(start, other_start)
            if shared > 0 and shared > 0.5 * min(end - start, other_end - other_start):
                overlap = True
                break
        if overlap:
            continue
        chosen.append((start, end))
        if len(chosen) >= max_windows:
            break
    return sorted(chosen) or [(0, min(limit, len(text)))]


def compose_relevant_excerpt(
    content: str,
    *,
    query: str = "",
    objective: str = "",
    char_budget: int = 6000,
    head_chars: int = 800,
    tail_chars: int = 400,
) -> tuple[str, list[dict]]:
    """Compose a bounded extractor view that keeps the task-relevant middle.

    ``clip_preserving_spans`` kept the head and tail, so a date in the body was lost
    whenever the page was longer than the extractor budget. Here the head and tail are
    still kept, but the remaining budget is spent on the window that best matches the
    task/query. Every omitted range is recorded explicitly.
    """
    text = content or ""
    if len(text) <= char_budget:
        return text, [{"start": 0, "end": len(text)}]
    head_chars = min(head_chars, max(40, char_budget // 8))
    tail_chars = min(tail_chars, max(20, char_budget // 12))
    middle_budget = max(120, char_budget - head_chars - tail_chars - 240)
    terms = _relevance_terms(query, objective)
    if not terms:
        # No task words to rank with: reuse the existing head/tail/exception window so
        # an unrelated long page still arrives with both ends and its caveats.
        from open_deep_research.context import clip_preserving_spans

        clipped = clip_preserving_spans(text, char_budget, head_ratio=0.5, tail_ratio=0.25)
        return clipped.text, [{"start": 0, "end": len(clipped.text), "compressed": True, "omitted_chars": clipped.omitted_chars}]
    windows = select_relevant_windows(
        text,
        query=query,
        objective=objective,
        limit=max(120, middle_budget // 2),
        max_windows=2,
    )
    per_window = max(120, middle_budget // max(1, len(windows)))
    head_end = min(head_chars, len(text))
    tail_start = max(head_end, len(text) - tail_chars)
    chunks: list[str] = [text[:head_end]]
    spans: list[dict] = [{"start": 0, "end": head_end}]
    cursor = head_end
    for win_start, win_end in windows:
        start = max(win_start, head_end)
        end = min(win_end, tail_start, start + per_window)
        if end - start < 120:
            continue
        if start > cursor:
            chunks.append(f"\n[... {start - cursor} characters omitted; next task-relevant section follows ...]\n")
        chunks.append(text[start:end])
        spans.append({"start": start, "end": end})
        cursor = max(cursor, end)
    if tail_start > cursor:
        chunks.append(f"\n[... {max(0, tail_start - cursor)} characters omitted; trailing exceptions follow ...]\n")
    chunks.append(text[tail_start:])
    spans.append({"start": tail_start, "end": len(text)})
    return "".join(chunks), spans


async def summarize_webpage_scoped(
    identity: Identity,
    store: Optional[ArtifactStore],
    summarization_model: Any,
    webpage_content: str,
    *,
    url: str = "",
    title: str = "",
    call_context: Any = None,
    config: RunnableConfig = None,
    query: str = "",
    objective: str = "",
    source_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Extract from one page with a recorded request and an archive-first guarantee.

    The original page is archived *before* the extractor runs. On failure or timeout
    only a bounded excerpt is returned (never the whole document), so a summarizer
    failure cannot silently flood the researcher's context.
    """
    from open_deep_research.context import ContextBuilder, assemble, call_context_for

    if call_context is None:
        # Legacy/direct callers: resolve from the caller's config. Empty config is a
        # declared gap, not a silent second default that differs from the run.
        call_context = call_context_for(config or {}, role="extractor", store=store)
        if config is None:
            obs.record_event("auxiliary_context_undeclared", role="extractor", reason_code="no_run_config")
    configurable = call_context.config
    # 400 chars stopped at the page's navigation in live runs: the Reviewer saw only
    # furniture while the claim quoted a line further down, and refused it as
    # unsupported. The excerpt is what a reviewer can see, so it must reach the body.
    snippet, excerpt_start, excerpt_end = select_relevant_window(
        webpage_content, query=query, objective=objective, limit=1200
    )
    evidence_id = ""
    if store is not None:
        try:
            record = store.add_evidence(
                identity,
                content=webpage_content,
                source_url=url,
                title=title,
                excerpt=snippet,
                locator={"excerpt_start": excerpt_start, "excerpt_end": excerpt_end, "excerpt_query": query or objective, **(source_metadata or {})},
                origin="direct_https_text" if source_metadata else "webpage_raw_content",
            )
            evidence_id = record.evidence_id
        except Exception:  # noqa: BLE001
            logging.warning("Could not archive webpage before extraction", exc_info=True)
    builder = ContextBuilder(configurable, store)
    messages, items = builder.extractor_messages(url, title, webpage_content, query=query, objective=objective)
    packet = builder.build(
        identity,
        "extractor",
        messages,
        items=items,
        model=configurable.summarization_model,
        declared_max_tokens=configurable.summarization_model_max_tokens,
        response_schema=Summary,
    )
    # Fit the page to the extractor budget before sending. The role rules and the
    # request scaffolding are charged to the same budget as the page, so a page clipped
    # to exactly ``budget * 4`` chars still estimated 4.1k-5.4k against a 4k cap and the
    # request was refused: the caller fell back to a Tavily snippet and the archive
    # never got a usable excerpt (every live acceptance run hit this). Shrink the page
    # until the whole request fits, reselecting windows from the original archive
    # each time so its task-relevant middle and original offsets remain available.
    view_cap = min(len(webpage_content), max(2000, builder.budget_tokens_for("extractor") * 4))
    for _ in range(5):
        if packet.estimated_input_tokens <= packet.budget_tokens:
            break
        ratio = packet.estimated_input_tokens / max(1, packet.budget_tokens)
        target = max(200, int(view_cap / ratio * 0.7))
        if target >= view_cap:
            break
        view_cap = target
        messages, items = builder.extractor_messages(url, title, webpage_content, query=query, objective=objective, char_budget=view_cap)
        packet = builder.build(
            identity,
            "extractor",
            messages,
            items=items,
            model=configurable.summarization_model,
            declared_max_tokens=configurable.summarization_model_max_tokens,
            response_schema=Summary,
        )
    pointer = f" [evidence: {evidence_id}]" if evidence_id else ""
    extraction_key = hashlib.sha256(json.dumps({
        "messages": [message.model_dump(mode="json") for message in packet.messages],
        "model": configurable.summarization_model, "schema": Summary.model_json_schema(),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    if store is not None and evidence_id:
        cached = store.cached_extraction(identity, evidence_id, extraction_key)
        if cached is not None:
            store.append_event({"type": "extraction_reused", "evidence_id": evidence_id, "request_hash": extraction_key})
            return cached
    extract_span = obs.span("page.extract", kind=obs.SPAN_STAGE, attributes={"channel": "scoped_extract", "role": "extractor", "model": configurable.summarization_model})
    try:
        with extract_span:
            summary = await asyncio.wait_for(
                assemble(
                    summarization_model,
                    packet,
                    store=store if configurable.artifacts_enabled else None,
                    record_requests=configurable.record_requests,
                    max_tokens=configurable.summarization_model_max_tokens,
                    model_name=configurable.summarization_model,
                    retries=configurable.max_structured_output_retries,
                    budget=call_context.budget,
                ),
                timeout=60.0,
            )
    except asyncio.TimeoutError:
        extract_span.fail(error_class="TimeoutError", status=obs.STATUS_TIMEOUT)
        logging.warning("Summarization timed out; returning bounded excerpt")
        return f"<summary>\n[extraction timed out] Verbatim excerpt: {snippet}\n</summary>{pointer}"
    except Exception as exc:  # noqa: BLE001
        extract_span.fail(error_class=type(exc).__name__)
        logging.warning("Summarization failed (%s); returning bounded excerpt", exc)
        return f"<summary>\n[extraction failed: {classify_tool_error(exc)}] Verbatim excerpt: {snippet}\n</summary>{pointer}"
    if summary is None or not getattr(summary, "summary", "").strip():
        return f"<summary>\n[empty extraction] Verbatim excerpt: {snippet}\n</summary>{pointer}"
    result = f"<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>{pointer}"
    if store is not None and evidence_id:
        store.cache_extraction(evidence_id, extraction_key, result)
    return result


def _has_publication_metadata(text: str) -> bool:
    cleaned = re.sub(r"[*_`]", "", text or "")
    return bool(re.search(r"(?:posted on|published (?:on|at)|release date|发布日期|发布于)\s*[:：]?\s*(?:\d{4}[-年/]\d{1,2}|[A-Z][a-z]+\s+\d{1,2})", cleaned, flags=re.IGNORECASE))


def _page_fetch_allowed(url: str, config: Configuration) -> bool:
    try:
        parts = urlsplit(url)
        path_allowed = not config.page_fetch_path_prefixes or any(parts.path.startswith(prefix) and parts.path.rstrip("/") != prefix.rstrip("/") for prefix in config.page_fetch_path_prefixes)
        return parts.scheme.lower() == "https" and not parts.username and not parts.password and parts.port in (None, 443) and path_allowed and any(official_domain_match(url, domain) for domain in config.page_fetch_domains)
    except ValueError:
        return False


async def fetch_missing_publication_metadata(
    result: Dict[str, Any], *, identity: Identity, store: Optional[ArtifactStore], config: Configuration,
) -> Optional[Dict[str, Any]]:
    """Read one allowed HTTPS page when search omitted its publication header.

    This is opt-in and run-bounded. Redirects stay inside the allowed domains, HTML
    bytes are archived, and parsed text remains a separate content-hashed snapshot.
    No model or search-provider request is made by this fallback.
    """
    url = str(result.get("url") or "")
    if store is None or not config.max_total_page_fetches or not _page_fetch_allowed(url, config) or _has_publication_metadata(str(result.get("raw_content") or "")):
        return None
    request_id = store.reserve_page_fetch(url, config.max_total_page_fetches)
    if request_id is None:
        return None
    fragment = None
    if str(result.get("raw_content") or "").strip():
        fragment = store.add_evidence(identity, content=str(result["raw_content"]), source_url=url,
                                      title=str(result.get("title") or ""), origin="search_raw_fragment")
    requests = 0
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            for _ in range(3):
                requests += 1
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        target = urljoin(url, response.headers.get("location", ""))
                        if not _page_fetch_allowed(target, config):
                            raise ValueError("page redirect leaves the allowed HTTPS domains")
                        url = target
                        continue
                    response.raise_for_status()
                    if "html" not in response.headers.get("content-type", "").lower():
                        raise ValueError("page is not HTML")
                    chunks = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 500_000:
                            raise ValueError("page exceeds the 500000-byte bound")
                        chunks.append(chunk)
                    html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                    break
            else:
                raise ValueError("page redirect limit exhausted")
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.title or soup.find("h1")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        published = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("meta", attrs={"itemprop": "datePublished"})
        date_text = str(published.get("content") or "") if published else ""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        body = soup.find("main") or soup.find("article") or soup.body or soup
        text = f"Title: {title}\n" + (f"Published at: {date_text}\n" if date_text else "") + body.get_text("\n", strip=True)
        raw = store.archive_document(identity, html, url=url, title=title)
        store.append_ledger("page_fetch_usage", {"request_id": request_id, "phase": "finished", "status": "archived", "http_requests": requests,
                                               "raw_html_hash": raw["content_hash"], "text_hash": hashlib.sha256(text.encode()).hexdigest()})
        return {**result, "url": url, "title": title, "raw_content": text,
                "source_metadata": {"raw_html_hash": raw["content_hash"], "parser": "beautifulsoup-html.parser", "transport": "direct_https",
                                    "supersedes_evidence_id": fragment.evidence_id if fragment is not None else None}}
    except Exception as exc:  # noqa: BLE001 - preserve the search fragment on failure
        store.append_ledger("page_fetch_usage", {"request_id": request_id, "phase": "finished", "status": "failed", "http_requests": requests, "error_class": type(exc).__name__})
        return None


async def evidence_aware_web_search(
    queries: List[str],
    *,
    identity: Identity,
    store: Optional[ArtifactStore],
    config: RunnableConfig,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> ToolResult:
    """Search + archive originals + extract with evidence ids (scoped mode).

    The raw page is stored before any model sees it, each source gets a stable
    ``evidence_id`` and the returned text points back to it.
    """
    configurable = Configuration.from_runnable_config(config)
    try:
        search_results = await tavily_search_async(
            queries,
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        error_class = classify_tool_error(exc)
        return ToolResult(
            status="error",
            error_class=error_class,
            error=f"{type(exc).__name__}: {exc}"[:500],
            content=f"Search failed ({error_class}). No results were returned; this does not mean the information does not exist.",
        )
    unique_results: Dict[str, Dict[str, Any]] = {}
    for response in search_results:
        for result in response.get("results", []):
            url = result.get("url", "")
            if url and url not in unique_results:
                unique_results[url] = {**result, "query": response.get("query", "")}

    # Spend the single fetch on a task-relevant publication, not the first homepage
    # returned by search. Ties prefer the shorter publication URL.
    terms = _relevance_terms(" ".join(queries), "")
    candidates = [row for row in unique_results.values() if _page_fetch_allowed(str(row.get("url") or ""), configurable) and not _has_publication_metadata(str(row.get("raw_content") or ""))]
    if candidates:
        priority_paths = {path.rstrip("/") for path in configurable.page_fetch_priority_paths}
        target = max(candidates, key=lambda row: (urlsplit(str(row.get("url") or "")).path.rstrip("/") in priority_paths, sum(term in (str(row.get("title") or "") + " " + str(row.get("url") or "")).lower() for term in terms), -len(str(row.get("url") or ""))))
        fetched = await fetch_missing_publication_metadata(target, identity=identity, store=store, config=configurable)
        if fetched is not None:
            target.update(fetched)

    from open_deep_research.context import call_context_for

    extract_context = call_context_for(config, role="extractor", store=store)
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    # Pass the raw chat model. ``assemble`` binds the response schema and performs
    # the bounded retry itself, so every attempt is reserved, counted and priced;
    # wrapping here would hide provider retries from the run ledger.
    # ``api_key=None`` is omitted for the same reason as in summarize_webpage: it would
    # override the provider SDK's environment lookup.
    summarization_kwargs: Dict[str, Any] = {
        "model": configurable.summarization_model,
        "max_tokens": configurable.summarization_model_max_tokens,
        "tags": ["langsmith:nostream"],
    }
    if model_api_key:
        summarization_kwargs["api_key"] = model_api_key
    options = provider_request_options(configurable.summarization_model)
    if options:
        summarization_kwargs["extra_body"] = options
    summarization_model = init_chat_model(**summarization_kwargs)

    evidence_ids: List[str] = []
    summaries: List[Optional[str]] = []
    for result in unique_results.values():
        raw = result.get("raw_content") or ""
        if not raw:
            summaries.append(None)
            continue
        text = await summarize_webpage_scoped(
            identity,
            store,
            summarization_model,
            raw,
            url=result.get("url", ""),
            title=result.get("title", ""),
            call_context=extract_context,
            query=str(result.get("query") or " ".join(queries)),
            **({"source_metadata": result["source_metadata"]} if result.get("source_metadata") else {}),
        )
        evidence_ids.extend(re.findall(r"\[evidence: (E\d+)\]", text))
        summaries.append(text)

    if not unique_results:
        return ToolResult(
            status="ok",
            content="No valid search results found. Please try different search queries or use a different search API.",
        )
    formatted = "Search results: \n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        content = result.get("content") if summaries[i] is None else summaries[i]
        formatted += f"\n\n--- SOURCE {i+1}: {result.get('title', '')} ---\n"
        formatted += f"URL: {url}\n\n"
        formatted += f"SUMMARY:\n{content}\n\n"
        formatted += "\n\n" + "-" * 80 + "\n"
    return ToolResult(status="ok", content=formatted, evidence_ids=evidence_ids)

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def provider_request_options(model_name: str) -> Dict[str, Any]:
    """Return the provider request options this deployment must set explicitly.

    DeepSeek enables thinking mode by default. Thinking mode rejects the forced
    ``tool_choice`` that structured output uses ("Thinking mode does not support this
    tool_choice", reproduced against ``deepseek-flash``), and its reasoning tokens are
    billed as output. The documented switch is ``{"thinking": {"type": "disabled"}}``
    (https://api-docs.deepseek.com/guides/thinking_mode). Other providers get no extra
    body: an unknown field would be rejected by their API.
    """
    provider = str(model_name or "").split(":", 1)[0].strip().lower()
    if provider == "deepseek":
        return {"thinking": {"type": "disabled"}}
    return {}


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        elif model_name.startswith("deepseek"):
            return api_keys.get("DEEPSEEK_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        elif model_name.startswith("deepseek"):
            return os.getenv("DEEPSEEK_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")


def bind_search_identity(tools: List[Any], identity: Identity, store: Optional[ArtifactStore]) -> List[Any]:
    """Attach the host identity to search tools for the duration of one request.

    The attribute is written by the host on a per-request copy, so a research task can
    never inherit the identity of another task (or of the supervisor).
    """
    bound: List[Any] = []
    for tool in tools:
        name = tool_name(tool)
        if name == "tavily_search" and isinstance(tool, BaseTool):
            clone = tool.model_copy(deep=False)
            clone._odr_identity = identity  # type: ignore[attr-defined]
            clone._odr_store = store  # type: ignore[attr-defined]
            bound.append(clone)
        else:
            bound.append(tool)
    return bound
