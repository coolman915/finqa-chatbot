"""LangSmith custom tracing callbacks for per-agent observability."""

from __future__ import annotations

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler


class FinQATracingCallback(BaseCallbackHandler):
    """Attach FinQA-specific metadata to LangSmith traces.

    Usage: pass as a callback when invoking the graph::

        graph.invoke(state, config={"callbacks": [FinQATracingCallback(entry_id="...", round_number=1)]})
    """

    def __init__(self, entry_id: str = "", round_number: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.entry_id = entry_id
        self.round_number = round_number

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata", {})
        metadata["entry_id"] = self.entry_id
        metadata["round_number"] = self.round_number
        kwargs["metadata"] = metadata
