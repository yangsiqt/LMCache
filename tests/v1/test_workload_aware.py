# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from lmcache.integration.vllm.workload_aware import (
    WorkloadAwareRequest,
    WorkloadAwareResultTracker,
    decide_retrieve,
)


def controls(mode: str, threshold: int | None = None) -> dict:
    workload_aware = {
        "retrieve_mode": mode,
        "request_id": "request-1",
        "session_id": "session-1",
        "trace_id": "request-1:0",
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
