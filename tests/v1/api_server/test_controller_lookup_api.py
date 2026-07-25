# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import AsyncMock

# Third Party
from fastapi.testclient import TestClient

# First Party
from lmcache.v1.api_server.__main__ import create_app
from lmcache.v1.cache_controller.message import LookupRetMsg


def test_lookup_http_response_preserves_revisioned_multi_tier_layout() -> None:
    app = create_app(
        {"pull": "127.0.0.1:19101", "reply": "127.0.0.1:19102"},
        health_check_interval=-1,
        lmcache_worker_timeout=300,
    )
    manager = app.state.lmcache_controller_manager
    manager.handle_orchestration_message = AsyncMock(
        return_value=LookupRetMsg(
            event_id="lookup-v2-2",
            layout_info={"instance": ("LocalCPUBackend", 512)},
            layout_info_v2=[
                ("instance", 0, "LocalCPUBackend", 512),
                ("instance", 0, "RemoteBackend", 256),
            ],
            layout_info_v3=[
                ("instance", 0, "LocalCPUBackend", 512, 7),
                ("instance", 0, "RemoteBackend", 256, 5),
            ],
        )
    )

    response = TestClient(app).post("/lookup", json={"tokens": [1, 2, 3]})

    assert response.status_code == 200
    assert response.json()["layout_info_v3"] == [
        ["instance", 0, "LocalCPUBackend", 512, 7],
        ["instance", 0, "RemoteBackend", 256, 5],
    ]
