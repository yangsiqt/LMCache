# SPDX-License-Identifier: Apache-2.0
"""Per-request retrieval controls used by workload-aware routers."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

RetrieveMode = Literal["auto", "force", "skip"]
_VALID_RETRIEVE_MODES = {"auto", "force", "skip"}


@dataclass(frozen=True)
class WorkloadAwareRequest:
    retrieve_mode: RetrieveMode = "auto"
    min_retrieve_tokens: int | None = None
    request_id: str = ""
    session_id: str = ""
    trace_id: str = ""

    @classmethod
    def from_kv_transfer_params(
        cls, params: dict[str, Any] | None
    ) -> "WorkloadAwareRequest | None":
        if not isinstance(params, dict):
            return None
        raw = params.get("workload_aware")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("kv_transfer_params.workload_aware must be an object")

        mode = raw.get("retrieve_mode", "auto")
        if mode not in _VALID_RETRIEVE_MODES:
            raise ValueError(f"invalid workload-aware retrieve_mode: {mode}")
        threshold = raw.get("min_retrieve_tokens")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int):
                raise ValueError("min_retrieve_tokens must be an integer")
            if threshold < 0:
                raise ValueError("min_retrieve_tokens must be non-negative")
        return cls(
            retrieve_mode=mode,
            min_retrieve_tokens=threshold,
            request_id=str(raw.get("request_id", "")),
            session_id=str(raw.get("session_id", "")),
            trace_id=str(raw.get("trace_id", "")),
        )


@dataclass(frozen=True)
class RetrieveDecision:
    should_retrieve: bool
    min_retrieve_tokens: int
    reason: str


def decide_retrieve(
    control: WorkloadAwareRequest,
    global_min_retrieve_tokens: int,
    available_tokens: int,
) -> RetrieveDecision:
    available_tokens = max(0, available_tokens)
    if control.retrieve_mode == "skip":
        return RetrieveDecision(False, 0, "router_skip")
    if available_tokens <= 0:
        return RetrieveDecision(False, 0, "external_miss")
    if control.retrieve_mode == "force":
        return RetrieveDecision(True, 0, "router_force")

    threshold = (
        control.min_retrieve_tokens
        if control.min_retrieve_tokens is not None
        else max(0, global_min_retrieve_tokens)
    )
    if available_tokens < threshold:
        return RetrieveDecision(False, threshold, "below_request_threshold")
    return RetrieveDecision(True, threshold, "request_threshold_met")


@dataclass
class WorkloadAwareResult:
    request_id: str
    session_id: str
    trace_id: str
    retrieve_mode: str
    actual_kv_path: str = "pending"
    vllm_cached_tokens: int = 0
    lmcache_cached_tokens: int = 0
    transfer_bytes: int | None = None
    lookup_ms: float | None = None
    load_ms: float | None = None
    deserialize_ms: float | None = None
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkloadAwareResultTracker:
    """Thread-safe lifecycle state shared by scheduler callbacks."""

    _STOP = object()

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        trace_path: str | Path | None = None,
    ) -> None:
        self._clock = clock
        self._results: dict[str, WorkloadAwareResult] = {}
        self._lookup_started: dict[str, float] = {}
        self._lock = threading.Lock()
        configured_path = trace_path or os.getenv("LMCACHE_WORKLOAD_AWARE_TRACE_PATH")
        self._trace_path = Path(configured_path) if configured_path else None
        self._trace_queue: queue.Queue[dict[str, Any] | object] | None = None
        self._trace_thread: threading.Thread | None = None
        if self._trace_path is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_queue = queue.Queue(maxsize=10000)
            self._trace_thread = threading.Thread(
                target=self._write_traces,
                name="lmcache-workload-trace",
                daemon=True,
            )
            self._trace_thread.start()

    def _write_traces(self) -> None:
        assert self._trace_path is not None and self._trace_queue is not None
        with self._trace_path.open("a", encoding="utf-8") as handle:
            while True:
                row = self._trace_queue.get()
                try:
                    if row is self._STOP:
                        return
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                finally:
                    self._trace_queue.task_done()

    def _trace(self, result: dict[str, Any]) -> None:
        if self._trace_queue is None:
            return
        try:
            self._trace_queue.put_nowait(result)
        except queue.Full:
            # The result is still returned to vLLM; trace pressure must not fail serving.
            pass

    def begin(
        self,
        request_id: str,
        control: WorkloadAwareRequest,
        vllm_cached_tokens: int,
    ) -> WorkloadAwareResult:
        with self._lock:
            result = self._results.get(request_id)
            if result is None:
                result = WorkloadAwareResult(
                    request_id=control.request_id or request_id,
                    session_id=control.session_id,
                    trace_id=control.trace_id,
                    retrieve_mode=control.retrieve_mode,
                )
                self._results[request_id] = result
            result.vllm_cached_tokens = max(0, vllm_cached_tokens)
            return result

    def start_lookup(self, request_id: str) -> None:
        with self._lock:
            self._lookup_started.setdefault(request_id, self._clock())

    def record_decision(
        self,
        request_id: str,
        decision: RetrieveDecision,
        lmcache_cached_tokens: int,
    ) -> None:
        with self._lock:
            result = self._results.get(request_id)
            if result is None:
                return
            started = self._lookup_started.pop(request_id, None)
            if started is not None:
                result.lookup_ms = max(0.0, (self._clock() - started) * 1000.0)
            result.lmcache_cached_tokens = max(0, lmcache_cached_tokens)
            if decision.should_retrieve:
                result.actual_kv_path = "lmcache_external"
                result.fallback_reason = ""
            else:
                result.actual_kv_path = (
                    "local_hbm" if result.vllm_cached_tokens > 0 else "recompute"
                )
                result.fallback_reason = decision.reason

    def record_scheduled_load(self, request_id: str, loaded_tokens: int) -> None:
        with self._lock:
            result = self._results.get(request_id)
            if result is None:
                return
            if loaded_tokens <= 0 and result.actual_kv_path == "lmcache_external":
                result.actual_kv_path = (
                    "local_hbm" if result.vllm_cached_tokens > 0 else "recompute"
                )
                result.fallback_reason = "scheduler_declined_load"

    def finish(self, request_id: str, aborted: bool = False) -> dict[str, Any] | None:
        with self._lock:
            self._lookup_started.pop(request_id, None)
            result = self._results.pop(request_id, None)
            if result is None:
                return None
            if aborted:
                result.fallback_reason = result.fallback_reason or "request_aborted"
            output = result.to_dict()
        self._trace(output)
        return output

    def close(self) -> None:
        if self._trace_queue is None or self._trace_thread is None:
            return
        self._trace_queue.join()
        self._trace_queue.put(self._STOP)
        self._trace_thread.join(timeout=5)
        self._trace_queue = None
        self._trace_thread = None
