"""
Langfuse tracing integration for RevRAG.

Wraps the Langfuse SDK to provide easy tracing of retrieval pipeline runs.
All traces include: query, retrieved passages, latency breakdown, and metrics.

Usage:
    tracer = LangfuseTracer(cfg)
    with tracer.trace("retrieve", query_id=qid, query=query) as span:
        results = revrag.retrieve(qid, query, tracer=span)
        span.update(metrics={"ndcg": ndcg_score})

Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars before running.
Set langfuse.enabled: false in config to disable without code changes.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("langfuse not installed. Tracing disabled.")


class _NoOpSpan:
    """No-op span used when Langfuse is disabled or unavailable."""
    def start_span(self, name, **kwargs):
        return self
    def end(self, **kwargs):
        pass
    def update(self, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class LangfuseTracer:
    """
    Thin wrapper around Langfuse for RevRAG pipeline tracing.

    Args:
        cfg: full config dict (reads langfuse section)
    """

    def __init__(self, cfg: Dict):
        lf_cfg = cfg.get("langfuse", {})
        self.enabled = lf_cfg.get("enabled", False) and LANGFUSE_AVAILABLE
        self.trace_name = lf_cfg.get("trace_name", "revrag_retrieval")
        self._client: Optional[Any] = None

        if self.enabled:
            public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", lf_cfg.get("public_key", ""))
            secret_key = os.environ.get("LANGFUSE_SECRET_KEY", lf_cfg.get("secret_key", ""))
            host = lf_cfg.get("host", "https://cloud.langfuse.com")
            if not public_key or not secret_key:
                logger.warning(
                    "Langfuse keys not set. Disabling tracing. "
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars."
                )
                self.enabled = False
            else:
                try:
                    self._client = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host,
                    )
                    logger.info("Langfuse tracing enabled.")
                except Exception as e:
                    logger.warning(f"Langfuse init failed: {e}. Tracing disabled.")
                    self.enabled = False

    @contextmanager
    def trace(self, operation: str, **metadata):
        """Context manager for tracing a retrieval operation."""
        if not self.enabled or self._client is None:
            yield _NoOpSpan()
            return

        trace = self._client.trace(
            name=f"{self.trace_name}.{operation}",
            metadata=metadata,
        )
        span = trace.span(name=operation)
        t0 = time.time()
        try:
            yield span
        finally:
            span.end(
                metadata={**metadata, "elapsed_ms": (time.time() - t0) * 1000}
            )
            self._client.flush()

    def start_span(self, name: str, **kwargs) -> _NoOpSpan:
        """Non-context-manager span (for use inside retrieval functions)."""
        if not self.enabled or self._client is None:
            return _NoOpSpan()
        # Return a minimal span object
        return _NoOpSpan()  # simplified; expand with real Langfuse span if needed

    def log_metrics(self, run_id: str, metrics: Dict) -> None:
        """Log a metrics dict to a Langfuse score event."""
        if not self.enabled or self._client is None:
            return
        for key, value in metrics.items():
            try:
                self._client.score(
                    trace_id=run_id,
                    name=key,
                    value=float(value),
                )
            except Exception as e:
                logger.debug(f"Langfuse score log failed: {e}")
