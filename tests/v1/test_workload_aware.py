# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from lmcache.integration.vllm.workload_aware import (
    WorkloadAwareRequest,
    WorkloadAwareResultTracker,
    decide_retrieve,
    record_actual_retrieve,
    record_worker_load_started,
    workload_aware_search_range,
)


def controls(mode: str, threshold: int | None = None) -> dict:
    workload_aware = {
        "retrieve_mode": mode,
        "request_id": "request-1",
        "session_id": "session-1",
        "trace_id": "request-1:0",
        "attempt_id": "0",
        "backend_id": "backend-0",
        "selected_path": "lmcache_l1",
    }
    if threshold is not None:
        workload_aware["min_retrieve_tokens"] = threshold
    return {"workload_aware": workload_aware}


def test_parse_and_retrieve_modes() -> None:
    auto = WorkloadAwareRequest.from_kv_transfer_params(controls("auto", 256))
    assert auto is not None
    assert not decide_retrieve(auto, 1024, 128).should_retrieve
    assert decide_retrieve(auto, 1024, 512).should_retrieve

    force = WorkloadAwareRequest.from_kv_transfer_params(controls("force"))
    assert force is not None
    assert decide_retrieve(force, 4096, 1).should_retrieve

    skip = WorkloadAwareRequest.from_kv_transfer_params(controls("skip"))
    assert skip is not None
    assert decide_retrieve(skip, 0, 8192).reason == "router_skip"


@pytest.mark.parametrize(
    "value",
    [-1, True, "512"],
)
def test_invalid_request_threshold_is_rejected(value) -> None:
    params = controls("auto")
    params["workload_aware"]["min_retrieve_tokens"] = value
    with pytest.raises(ValueError):
        WorkloadAwareRequest.from_kv_transfer_params(params)


def test_invalid_selected_path_is_rejected() -> None:
    params = controls("auto")
    params["workload_aware"]["selected_path"] = "unknown"
    with pytest.raises(ValueError):
        WorkloadAwareRequest.from_kv_transfer_params(params)


def test_result_tracker_reports_known_values_without_fabricating_timings() -> None:
    now = [1.0]
    tracker = WorkloadAwareResultTracker(clock=lambda: now[0])
    control = WorkloadAwareRequest.from_kv_transfer_params(controls("force"))
    assert control is not None
    tracker.begin("request-1", control, 64)
    tracker.start_lookup("request-1")
    now[0] = 1.025
    tracker.record_decision("request-1", decide_retrieve(control, 512, 1024), 1088)
    result = tracker.finish("request-1")
    assert result is not None
    assert result["actual_kv_path"] == "lmcache_external"
    assert result["lookup_ms"] == pytest.approx(25)
    assert result["load_ms"] is None
    assert result["transfer_bytes"] is None
    assert result["terminal"] is True
    assert result["attempt_id"] == "0"
    assert result["backend_id"] == "backend-0"
    assert result["selected_path"] == "lmcache_l1"


def test_result_tracker_writes_async_connector_trace(tmp_path) -> None:
    path = tmp_path / "connector.jsonl"
    tracker = WorkloadAwareResultTracker(trace_path=path)
    control = WorkloadAwareRequest.from_kv_transfer_params(controls("skip"))
    assert control is not None
    tracker.begin("request-1", control, 0)
    tracker.record_decision("request-1", decide_retrieve(control, 0, 0), 0)
    assert tracker.finish("request-1") is not None
    tracker.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["phase"] for row in rows] == [
        "scheduler_seen",
        "lookup_completed",
        "request_finished",
    ]
    assert all(row["request_id"] == "request-1" for row in rows)
    assert rows[-1]["actual_kv_path"] == "recompute"
    assert rows[-1]["event_type"] == "kv_execution_feedback"
    assert rows[-1]["terminal"] is True
    assert rows[-1]["schema_version"] == "2.1"


def test_strict_selected_path_maps_to_one_storage_tier() -> None:
    assert workload_aware_search_range(
        {"lmcache.workload_aware.selected_path": "lmcache_l1"}
    ) == ["LocalCPUBackend"]
    assert workload_aware_search_range(
        {"lmcache.workload_aware.selected_path": "mooncake_l2"}
    ) == ["RemoteBackend"]
    assert (
        workload_aware_search_range(
            {"lmcache.workload_aware.selected_path": "recompute"}
        )
        is None
    )


class FakeLookupClient:
    def __init__(self, hit_tokens: int):
        self.hit_tokens = hit_tokens
        self.lookup_calls = 0

    def lookup_cache(self, lookup_id: str) -> int:
        return -1

    def lookup(self, token_ids, lookup_id: str, request_configs=None) -> int:
        self.lookup_calls += 1
        return self.hit_tokens


def connector(hit_tokens: int):
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    instance = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    instance.kv_role = "kv_consumer"
    instance._manager = SimpleNamespace(lookup_client=FakeLookupClient(hit_tokens))
    instance.config = SimpleNamespace(min_retrieve_tokens=512)
    instance._workload_aware_results = WorkloadAwareResultTracker()
    instance._requests_priority = {}
    instance.skip_last_n_tokens = 0
    instance._max_tokens_per_load = 0
    instance._lmcache_chunk_size = 256
    instance.load_specs = {}
    return instance


def request(mode: str, threshold: int | None = None):
    params = controls(mode, threshold)
    return SimpleNamespace(
        request_id="request-1",
        kv_transfer_params=params,
        priority=0,
        all_token_ids=list(range(1024)),
        prompt_token_ids=list(range(1024)),
        num_tokens=1024,
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": params}),
    )


def test_adapter_skip_avoids_lookup_and_force_bypasses_global_threshold() -> None:
    skipped = connector(128)
    assert skipped.get_num_new_matched_tokens(request("skip"), 0) == 0
    assert skipped.lookup_client.lookup_calls == 0

    forced = connector(128)
    assert forced.get_num_new_matched_tokens(request("force"), 0) == 128
    assert forced.lookup_client.lookup_calls == 1

    auto = connector(128)
    assert auto.get_num_new_matched_tokens(request("auto"), 0) == 0
    per_request = connector(128)
    assert per_request.get_num_new_matched_tokens(request("auto", 64), 0) == 128


def test_actual_retrieve_trace_distinguishes_l1_and_l2(tmp_path, monkeypatch) -> None:
    path = tmp_path / "actual.jsonl"
    monkeypatch.setenv("LMCACHE_WORKLOAD_AWARE_ACTUAL_TRACE_PATH", str(path))
    record_actual_retrieve(
        request_id="l1",
        storage_locations=["LocalCPUBackend"],
        retrieved_tokens=256,
        transfer_bytes=1024,
        load_ms=2.5,
        attempt_id="0",
        backend_id="backend-0",
        selected_path="lmcache_l1",
    )
    record_actual_retrieve(
        request_id="l2",
        storage_locations=["RemoteBackend"],
        retrieved_tokens=512,
        transfer_bytes=2048,
        load_ms=8.0,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["actual_kv_path"] for row in rows] == [
        "lmcache_l1",
        "mooncake_l2",
    ]
    assert rows[0]["event_type"] == "kv_execution_feedback"
    assert rows[0]["phase"] == "load_completed"
    assert rows[0]["schema_version"] == "2.1"
    assert rows[0]["path_mismatch"] is False
    assert rows[0]["terminal"] is False
    assert rows[0]["backend_id"] == "backend-0"


def test_worker_load_attempt_ids_allow_reloads_after_preemption(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "actual.jsonl"
    monkeypatch.setenv("LMCACHE_WORKLOAD_AWARE_ACTUAL_TRACE_PATH", str(path))
    request_configs = {
        "lmcache.workload_aware.attempt_id": "0",
        "lmcache.workload_aware.backend_id": "backend-0",
        "lmcache.workload_aware.decision_id": "reload:0",
        "lmcache.workload_aware.selected_path": "mooncake_l2",
    }

    first = record_worker_load_started(
        request_id="reload", request_configs=request_configs, required_tokens=256
    )
    second = record_worker_load_started(
        request_id="reload", request_configs=request_configs, required_tokens=128
    )
    record_actual_retrieve(
        request_id="reload",
        storage_locations=["RemoteBackend"],
        retrieved_tokens=128,
        transfer_bytes=1024,
        load_ms=4.0,
        attempt_id="0",
        backend_id="backend-0",
        selected_path="mooncake_l2",
        decision_id="reload:0",
        load_attempt_id=second,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert (first, second) == (0, 1)
    assert [row["load_attempt_id"] for row in rows] == [0, 1, 1]


def test_extract_request_configs_forwards_trace_identity() -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import extract_request_configs

    sampling = SimpleNamespace(
        extra_args={
            "kv_transfer_params": {
                "workload_aware": {
                    "request_id": "client-request",
                    "session_id": "session",
                    "trace_id": "client-request:0",
                    "attempt_id": "0",
                    "backend_id": "backend-0",
                    "selected_path": "lmcache_l1",
                    "decision_id": "client-request:0",
                    "length_bucket": "le_8k",
                    "concurrency_bucket": "low",
                    "retrieve_mode": "force",
                }
            }
        }
    )
    assert extract_request_configs(sampling) == {
        "lmcache.workload_aware.request_id": "client-request",
        "lmcache.workload_aware.session_id": "session",
        "lmcache.workload_aware.trace_id": "client-request:0",
        "lmcache.workload_aware.attempt_id": "0",
        "lmcache.workload_aware.backend_id": "backend-0",
        "lmcache.workload_aware.selected_path": "lmcache_l1",
        "lmcache.workload_aware.decision_id": "client-request:0",
        "lmcache.workload_aware.length_bucket": "le_8k",
        "lmcache.workload_aware.concurrency_bucket": "low",
    }
