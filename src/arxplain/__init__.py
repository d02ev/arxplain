"""ArXplain — explain a research paper at the reader's own level.

Multi-step agentic pipeline (ingest -> extract -> gap-analysis ->
planner -> explainer -> quiz) over a provider-agnostic LLM layer.
The primary interface is the CLI, writing a .md or .html explanation.
"""
from .models import (Dimension, ExplainResult, Level, Paper, Resource,
                     Topic, UserProfile)
from .pipeline import ArXplainPipeline, explain_paper
from .profiler import ProfilerAgent

__all__ = ["ArXplainPipeline", "Dimension", "ExplainResult", "Level",
           "Paper", "ProfilerAgent", "Resource", "Topic", "UserProfile",
           "explain_paper", "main"]
__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    from .cli import main as _cli_main
    return _cli_main(argv)
