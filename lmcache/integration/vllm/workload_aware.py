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
_VALID_SELECTED_PATHS = {
    "",
    "local_hbm",
    "lmcache_l1",
    "mooncake_l2",
    "recompute",
}

_STRICT_SEARCH_RANGES = {
    "lmcache_l1": ["LocalCPUBackend"],
    "mooncake_l2": ["RemoteBackend"],
}


def workload_aware_search_range(
    request_configs: dict[str, Any] | None,
) -> list[str] | None:
    """Return the strict storage search range selected by the Router."""
    if not request_configs:
        return None
    selected_path = str(request_configs.get("lmcache.workload_aware.selected_path", ""))
    search_range = _STRICT_SEARCH_RANGES.get(selected_path)
    return list(search_range) if search_range is not None else None


def _append_worker_event(row: dict[str, Any]) -> None:
    trace_path = os.getenv("LMCACHE_WORKLOAD_AWARE_ACTUAL_TRACE_PATH")
    if not trace_path or not row.get("request_id"):
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def record_worker_load_started(
    *,
    request_id: str,
    request_configs: dict[str, Any],
    required_tokens: int,
) -> None:
    _append_worker_event(
        {
            "schema_version": "2.1",
            "event_type": "kv_execution_feedback",
            "phase": "load_started",
            "terminal": False,
            "request_id": request_id,
            "attempt_id": str(
                request_configs.get("lmcache.workload_aware.attempt_id", "")
            ),
            "decision_id": str(
                request_configs.get("lmcache.workload_aware.decision_id", "")
            ),
            "backend_id": str(
                request_configs.get("lmcache.workload_aware.backend_id", "")
            ),
            "selected_path": str(
                request_configs.get("lmcache.workload_aware.selected_path", "")
            ),
            "length_bucket": str(
                request_configs.get("lmcache.workload_aware.length_bucket", "")
            ),
            "concurrency_bucket": str(
                request_configs.get("lmcache.workload_aware.concurrency_bucket", "")
            ),
            "required_tokens": max(0, int(required_tokens)),
            "recorded_at": time.time(),
        }
    )


def record_actual_retrieve(
    *,
    request_id: str,
    storage_locations: list[str],
    retrieved_tokens: int,
    transfer_bytes: int,
    load_ms: float,
    attempt_id: str = "",
    backend_id: str = "",
    selected_path: str = "",
    decision_id: str = "",
    length_bucket: str = "",
    concurrency_bucket: str = "",
    process_tokens_ms: float = 0.0,
    to_gpu_ms: float = 0.0,
    broadcast_ms: float = 0.0,
) -> None:
    """Append worker-observed retrieval evidence for experiment trace joins."""
    locations = sorted(set(storage_locations))
    if not locations or retrieved_tokens <= 0:
        actual_path = "recompute"
    elif locations == ["LocalCPUBackend"]:
        actual_path = "lmcache_l1"
    elif all(
        location == "RemoteBackend" or "mooncake" in location.lower()
        for location in locations
    ):
        actual_path = "mooncake_l2"
    else:
        actual_path = "mixed_external"
    row = {
        "schema_version": "2.1",
        "event_type": "kv_execution_feedback",
        "phase": "load_completed",
        "terminal": False,
        "request_id": request_id,
        "attempt_id": attempt_id,
        "backend_id": backend_id,
        "selected_path": selected_path,
        "decision_id": decision_id,
        "length_bucket": length_bucket,
        "concurrency_bucket": concurrency_bucket,
        "actual_kv_path": actual_path,
        "path_mismatch": bool(
            selected_path in {"lmcache_l1", "mooncake_l2"}
            and actual_path != selected_path
        ),
        "storage_locations": locations,
        "retrieved_tokens": max(0, int(retrieved_tokens)),
        "transfer_bytes": max(0, int(transfer_bytes)),
        "load_ms": max(0.0, float(load_ms)),
        "process_tokens_ms": max(0.0, float(process_tokens_ms)),
        "to_gpu_ms": max(0.0, float(to_gpu_ms)),
        "broadcast_ms": max(0.0, float(broadcast_ms)),
        "recorded_at": time.time(),
    }
    _append_worker_event(row)


@dataclass(frozen=True)
class WorkloadAwareRequest:
    retrieve_mode: RetrieveMode = "auto"
    min_retrieve_tokens: int | None = None
    request_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    attempt_id: str = ""
    backend_id: str = ""
    selected_path: str = ""
    decision_id: str = ""
    length_bucket: str = ""
    concurrency_bucket: str = ""

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
        selected_path = str(raw.get("selected_path", ""))
        if selected_path not in _VALID_SELECTED_PATHS:
            raise ValueError(f"invalid workload-aware selected_path: {selected_path}")
        return cls(
            retrieve_mode=mode,
            min_retrieve_tokens=threshold,
            request_id=str(raw.get("request_id", "")),
            session_id=str(raw.get("session_id", "")),
            trace_id=str(raw.get("trace_id", "")),
            attempt_id=str(raw.get("attempt_id", "")),
            backend_id=str(raw.get("backend_id", "")),
            selected_path=selected_path,
            decision_id=str(raw.get("decision_id", "")),
            length_bucket=str(raw.get("length_bucket", "")),
            concurrency_bucket=str(raw.get("concurrency_bucket", "")),
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
    attempt_id: str
    backend_id: str
    selected_path: str
    decision_id: str
    length_bucket: str
    concurrency_bucket: str
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
        self._emitted_phases: dict[str, set[str]] = {}
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
            # The result still reaches vLLM; trace pressure must not fail serving.
            pass

    def _trace_phase(
        self,
        request_id: str,
        phase: str,
        result: WorkloadAwareResult,
        **fields: Any,
    ) -> None:
        emitted = self._emitted_phases.setdefault(request_id, set())
        if phase in emitted:
            return
        emitted.add(phase)
        row = result.to_dict()
        row.update(
            {
                "schema_version": "2.1",
                "event_type": "kv_execution_feedback",
                "phase": phase,
                "terminal": False,
                "recorded_at": time.time(),
                **fields,
            }
        )
        self._trace(row)

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
                    attempt_id=control.attempt_id,
                    backend_id=control.backend_id,
                    selected_path=control.selected_path,
                    decision_id=control.decision_id
                    or f"{control.request_id or request_id}:{control.attempt_id}",
                    length_bucket=control.length_bucket,
                    concurrency_bucket=control.concurrency_bucket,
                    retrieve_mode=control.retrieve_mode,
                )
                self._results[request_id] = result
            result.vllm_cached_tokens = max(0, vllm_cached_tokens)
            self._trace_phase(request_id, "scheduler_seen", result)
            return result

    def start_lookup(self, request_id: str) -> None:
        with self._lock:
            self._lookup_started.setdefault(request_id, self._clock())
            result = self._results.get(request_id)
            if result is not None:
                self._trace_phase(request_id, "lookup_started", result)

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
            self._trace_phase(
                request_id,
                "lookup_completed",
                result,
                lookup_outcome=decision.reason,
            )

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
            self._trace_phase(
                request_id,
                "scheduler_admitted",
                result,
                scheduled_external_tokens=max(0, loaded_tokens),
            )

    def finish(self, request_id: str, aborted: bool = False) -> dict[str, Any] | None:
        with self._lock:
            self._lookup_started.pop(request_id, None)
            result = self._results.pop(request_id, None)
            if result is None:
                return None
            self._emitted_phases.pop(request_id, None)
            if aborted:
                result.fallback_reason = result.fallback_reason or "request_aborted"
            output = result.to_dict()
            output.update(
                {
                    "schema_version": "2.1",
                    "event_type": "kv_execution_feedback",
                    "phase": "request_finished",
                    "terminal": True,
                    "recorded_at": time.time(),
                }
            )
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
