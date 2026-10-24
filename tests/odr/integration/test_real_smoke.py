"""Optional real-provider smoke test. Skipped unless explicitly enabled.

Run it separately from the offline suite:

    RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest tests/odr/integration -q -s

It uses the models configured in ``.env`` (DeepSeek in this workspace) and the real
Tavily API, so it costs money and needs both ``TAVILY_API_KEY`` and a model key.
The offline suite never imports a provider.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from tests.odr.harness import run_graph

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_INTEGRATION") != "1",
    reason="set RUN_REAL_INTEGRATION=1 to run the real Tavily/LLM smoke test",
)


def test_real_search_review_and_report(artifact_root):
    load_dotenv()
    missing = [name for name in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY") if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing credentials: {', '.join(missing)}")

    result = run_graph(
        None,
        review_enabled=True,
        run_id="real-smoke",
        question="What is the official release date of Python 3.13? Answer with a source.",
        max_researcher_iterations=2,
        max_review_events=4,
        max_total_model_calls=12,
        max_total_tokens=200_000,
    )
    state = result.state
    assert state["final_report"].strip()
    assert state["run_status"] in {"complete", "partial"}
    # The run is observable even when it legitimately stops at a cap.
    assert (artifact_root / "real-smoke" / "metrics" / "summary.json").exists()