"""Configuration management for the Open Deep Research system."""

import os
from contextvars import ContextVar
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class ContextPolicy(str, Enum):
    """Context-assembly policy, the first experimental factor (A/B vs C/D).

    ``baseline`` keeps the original prompts, full history and full tool set.
    ``scoped`` routes every model call through the ContextBuilder.
    """

    BASELINE = "baseline"
    SCOPED = "scoped"

class StopPolicy(str, Enum):
    """How the run reacts to the global budget / iteration cap."""

    REVIEW = "review"
    PARTIAL = "partial"

class ExperimentArm(str, Enum):
    """The four experimental arms of the design documents."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


# Host-bound identity override. It is set by the runtime (never by a model) so that
# the web-summarizer / compressor calls, which run inside tools, keep the identity of
# the agent invocation that triggered them.
IDENTITY_CONTEXT: ContextVar[Optional[dict]] = ContextVar("odr_identity", default=None)

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    # Context / review features (docs/context-review-design). Both switches default
    # to the original behaviour so existing callers are unaffected.
    experiment_arm: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "",
                "description": "Optional convenience switch that sets both factors: A=baseline/no reviewer, B=scoped/no reviewer, C=baseline/reviewer, D=scoped/reviewer.",
                "options": [
                    {"label": "(explicit)", "value": ""},
                    {"label": "A: baseline, no reviewer", "value": "A"},
                    {"label": "B: scoped, no reviewer", "value": "B"},
                    {"label": "C: baseline, reviewer", "value": "C"},
                    {"label": "D: scoped, reviewer", "value": "D"},
                ],
            }
        },
    )
    context_policy: ContextPolicy = Field(
        default=ContextPolicy.BASELINE,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "baseline",
                "description": "baseline keeps the original context assembly; scoped enables the ContextBuilder (minimal sufficient context).",
                "options": [
                    {"label": "baseline", "value": "baseline"},
                    {"label": "scoped", "value": "scoped"},
                ],
            }
        },
    )
    reviewer_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable the independent Reviewer gates (plan / progress / readiness / report).",
            }
        },
    )
    artifact_root: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".odr_runs",
                "description": "Root directory for run artifacts (evidence, requests, context manifests, metrics). Relative paths resolve against the working directory.",
            }
        },
    )
    artifacts_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Persist evidence, request/context manifests and metrics. Observational only: never changes model input.",
            }
        },
    )
    record_requests: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Write the exact final visible request (credential-free) for every model call.",
            }
        },
    )
    # Reviewer budgets and limits (design starting points, not measured optima).
    reviewer_model: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "",
                "description": "Model for the Reviewer. Empty means use research_model (default for the first experiment).",
            }
        },
    )
    reviewer_model_max_tokens: int = Field(default=4000)
    max_review_events: int = Field(default=6)
    max_review_model_calls: int = Field(default=16)
    max_revision_rounds: int = Field(default=2)
    review_timeout_seconds: float = Field(default=180.0)
    # Global run budget.
    max_total_model_calls: int = Field(default=120)
    max_total_tokens: int = Field(default=2_000_000)
    budget_stop_policy: StopPolicy = Field(default=StopPolicy.REVIEW)
    # Context budgets (input soft caps, tokens).
    supervisor_context_budget: int = Field(default=6000)
    researcher_context_budget: int = Field(default=12000)
    reviewer_context_budget: int = Field(default=8000)
    writer_context_budget: int = Field(default=16000)
    compressor_context_budget: int = Field(default=6000)
    extractor_context_budget: int = Field(default=4000)
    brief_context_budget: int = Field(default=12000)


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {}
        for field_name in field_names:
            value = os.environ.get(field_name.upper(), configurable.get(field_name))
            # An explicit empty value (unset .env entry, cleared UI field) means "not
            # set"; it must not be coerced to "" and then to 0/False by pydantic.
            if isinstance(value, str) and not value.strip():
                value = None
            values[field_name] = value
        return cls(**{k: v for k, v in values.items() if v is not None})

    def apply_experiment_arm(self) -> "Configuration":
        """Resolve the explicit ``experiment_arm`` (A/B/C/D) over the two factors."""
        arm = os.environ.get("EXPERIMENT_ARM") or getattr(self, "experiment_arm", None)
        if not arm:
            return self
        arm = str(getattr(arm, "value", arm)).upper()
        mapping = {
            "A": (ContextPolicy.BASELINE, False),
            "B": (ContextPolicy.SCOPED, False),
            "C": (ContextPolicy.BASELINE, True),
            "D": (ContextPolicy.SCOPED, True),
        }
        if arm not in mapping:
            raise ValueError(f"Unknown experiment_arm {arm!r}; expected A, B, C or D")
        policy, reviewer = mapping[arm]
        return self.model_copy(update={"context_policy": policy, "reviewer_enabled": reviewer})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True