import importlib
import runpy
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient


@dataclass
class ModelFieldsCase:
    key: str
    payload: dict[str, Any]
    settings_field: str
    expected_value: Any


@pytest.fixture
def load_main(monkeypatch):
    def _load(**env_overrides):
        env = {
            "LISTEN_HOST": "127.0.0.1",
            "INPUT_PORT": "0",
            "OUTPUT_PORT": "59999",
            "PROCESSOR": "passthrough",
        }
        env.update(env_overrides)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        import main as module

        importlib.reload(module)
        return module

    return _load


def test_settings_page_serves_html(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


def test_health_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_status_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/status")
        assert response.status_code == 200
        body = response.json()
        assert body["processor"]["name"] == "passthrough"


def test_processors_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/processors")
        assert response.status_code == 200
        names = [entry["name"] for entry in response.json()["processors"]]
        assert "passthrough" in names


def test_get_runtime_settings_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/api/settings")
        assert response.status_code == 200
        assert response.json()["processor"] == "passthrough"


def test_patch_runtime_settings_with_no_changes_reports_unchanged(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.patch("/api/settings", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unchanged"
        assert body["processor_restarted"] is False


def test_patch_runtime_settings_applies_live_centre_reduction(load_main) -> None:
    main_module = load_main(PROCESSOR="stereo-centre-reduction")
    with TestClient(main_module.app) as client:
        response = client.patch(
            "/api/settings", json={"centre_reduction": {"reduction": 0.3}}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "applied"
        assert body["processor_restarted"] is False
        assert body["settings"]["centre_reduction"]["reduction"] == 0.3


@pytest.mark.parametrize(
    "case",
    [
        ModelFieldsCase(
            "demucs",
            {
                "model": "htdemucs_ft",
                "device": "cpu",
                "segment_seconds": 4.0,
                "overlap": 0.1,
                "shifts": 1,
                "vocal_reduction": 0.6,
            },
            "model",
            "htdemucs_ft",
        ),
        ModelFieldsCase(
            "convtasnet",
            {
                "model_path": "/models/other",
                "device": "cpu",
                "segment_seconds": 2.0,
                "vocal_reduction": 0.6,
                "vocal_source_index": 0,
                "accompaniment_source_index": 1,
            },
            "model_path",
            "/models/other",
        ),
        ModelFieldsCase(
            "mdx23c",
            {
                "device": "cpu",
                "segment_seconds": 1.5,
                "overlap": 0.1,
                "batch_size": 1,
                "vocal_reduction": 0.6,
                "precision": "float32",
            },
            "segment_seconds",
            1.5,
        ),
    ],
)
def test_patch_runtime_settings_changes_model_fields(
    load_main, case: ModelFieldsCase
) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.patch("/api/settings", json={case.key: case.payload})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "applied"
        assert body["processor_restarted"] is True
        assert body["settings"][case.key][case.settings_field] == case.expected_value


def test_patch_runtime_settings_rejects_semantically_invalid_update(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.patch("/api/settings", json={"processor": ""})
        assert response.status_code == 400
        assert "processor must not be empty" in response.json()["detail"]


def test_put_runtime_settings_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.put(
            "/api/settings", json={"centre_reduction": {"reduction": 0.4}}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "applied"


def test_delete_runtime_settings_restores_defaults(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        client.patch(
            "/api/settings",
            json={"demucs": {"segment_seconds": 3.0}},
        )
        response = client.delete("/api/settings")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "restored"
        assert body["settings"]["demucs"]["segment_seconds"] == pytest.approx(6.0)


def test_delete_runtime_settings_returns_400_on_failure(load_main, monkeypatch) -> None:
    main_module = load_main()

    async def boom() -> dict:
        raise ValueError("bad state")

    monkeypatch.setattr(main_module.service, "restore_startup_settings", boom)
    with TestClient(main_module.app) as client:
        response = client.delete("/api/settings")
        assert response.status_code == 400
        assert "bad state" in response.json()["detail"]


def test_reset_processor_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.post("/processor/reset")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["message"] == "Processor state reset"


def test_select_processor_endpoint_switches_processor(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.put("/processor/null")
        assert response.status_code == 200
        body = response.json()
        assert body["processor"] == "null"


def test_select_processor_endpoint_rejects_unknown_processor(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.put("/processor/does-not-exist")
        assert response.status_code == 400
        assert "Unknown processor" in response.json()["detail"]


def test_metrics_endpoint(load_main) -> None:
    main_module = load_main()
    with TestClient(main_module.app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "karaoke_anything_uptime_seconds" in response.text


def test_flatten_updates_only_includes_changed_fields(load_main) -> None:
    main_module = load_main()
    payload = main_module.RuntimeSettingsUpdate(
        processor="passthrough",
        demucs=main_module.DemucsSettingsUpdate(model=None, vocal_reduction=1.0),
        convtasnet=main_module.ConvTasNetSettingsUpdate(
            model_path=None, vocal_reduction=1.0
        ),
        centre_reduction=main_module.CentreReductionSettingsUpdate(reduction=0.7),
    )

    updates = main_module._flatten_updates(payload, main_module.service.settings)

    assert updates == {}


def test_flatten_updates_reports_changed_fields(load_main) -> None:
    main_module = load_main()
    payload = main_module.RuntimeSettingsUpdate(
        processor="null",
        demucs=main_module.DemucsSettingsUpdate(model="htdemucs_ft"),
        convtasnet=main_module.ConvTasNetSettingsUpdate(device="cpu"),
        centre_reduction=main_module.CentreReductionSettingsUpdate(reduction=0.2),
    )

    updates = main_module._flatten_updates(payload, main_module.service.settings)

    assert updates == {
        "processor": "null",
        "demucs_model": "htdemucs_ft",
        "convtasnet_device": "cpu",
        "centre_reduction": 0.2,
    }


def test_main_module_runs_as_script(monkeypatch) -> None:
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("LISTEN_HOST", "127.0.0.1")
    monkeypatch.setenv("INPUT_PORT", "0")
    monkeypatch.setenv("OUTPUT_PORT", "59999")
    monkeypatch.setenv("PROCESSOR", "passthrough")
    monkeypatch.delitem(sys.modules, "main", raising=False)

    runpy.run_module("main", run_name="__main__")

    assert len(calls) == 1
