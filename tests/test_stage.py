import pytest

from pipelines.stage import validate_stage_call


def test_valid_train_arguments():
    result = validate_stage_call(
        "src.train",
        ["--model", "random_forest", "--C", "1.0"],
    )
    assert result == ["--model", "random_forest", "--C", "1.0"]


def test_valid_dummy_model():
    result = validate_stage_call("src.train", ["--model", "dummy"])
    assert result == ["--model", "dummy"]


def test_valid_promote_run_id():
    run_id = "0123456789abcdef0123456789abcdef"
    result = validate_stage_call("src.promote", ["--run-id", run_id])
    assert result == ["--run-id", run_id]


def test_reject_unknown_module():
    with pytest.raises(ValueError):
        validate_stage_call("src.unknown", [])


def test_reject_unknown_model():
    with pytest.raises(ValueError):
        validate_stage_call("src.train", ["--model", "unknown"])


def test_reject_negative_c():
    with pytest.raises(ValueError):
        validate_stage_call("src.train", ["--C", "-1"])


def test_reject_filter_option():
    with pytest.raises(ValueError):
        validate_stage_call(
            "src.promote",
            ["--filter", "metrics.f1 > 0.8"],
        )


def test_reject_invalid_run_id():
    with pytest.raises(ValueError):
        validate_stage_call("src.promote", ["--run-id", "invalid"])


@pytest.mark.parametrize(
    "relative", ["../escape.csv", "/tmp/escape.csv", "data/../../escape", "data\\escape"]
)
def test_workspace_rejects_escaping_paths(tmp_path, relative):
    from pipelines.stage import workspace_path

    with pytest.raises(ValueError):
        workspace_path(tmp_path / "work", relative)


def test_workspace_rejects_symlink_escape(tmp_path):
    from pipelines.stage import workspace_path

    work = tmp_path / "work"
    work.mkdir()
    (work / "data").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace_path(work, "data/escape.csv")


def test_pull_rejects_traversal_before_s3_download(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pipelines import stage

    downloads = []
    client = SimpleNamespace(
        list_buckets=lambda: {"Buckets": [{"Name": "bucket"}]},
        get_paginator=lambda _: SimpleNamespace(
            paginate=lambda **_: [{"Contents": [{"Key": "run/../escape.csv"}]}]
        ),
        download_file=lambda *args: downloads.append(args),
    )
    monkeypatch.setattr(stage, "_s3", lambda _: (client, "bucket", "run"))
    with pytest.raises(ValueError):
        stage.pull("s3://bucket/run", tmp_path / "work")
    assert downloads == []
    assert not (tmp_path / "escape.csv").exists()


def test_pull_downloads_nested_artifact(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from pipelines import stage

    target = tmp_path / "work/data/raw.csv"
    client = SimpleNamespace(
        list_buckets=lambda: {"Buckets": [{"Name": "bucket"}]},
        get_paginator=lambda _: SimpleNamespace(
            paginate=lambda **_: [{"Contents": [{"Key": "run/data/raw.csv"}]}]
        ),
        download_file=lambda bucket, key, path: Path(path).write_text("csv"),
    )
    monkeypatch.setattr(stage, "_s3", lambda _: (client, "bucket", "run"))
    stage.pull("s3://bucket/run", tmp_path / "work")
    assert target.read_text() == "csv"


def test_xcom_content_is_json_data_not_a_destination(tmp_path, monkeypatch):
    import json
    import sys

    from pipelines import stage

    destination = tmp_path / "xcom/return.json"
    payload = {"mlflow_run_id": "../../outside", "metrics": {"text": "../escape"}}
    monkeypatch.setattr(stage, "XCOM_FILE", destination)
    monkeypatch.setattr(stage, "pull", lambda *args: None)
    monkeypatch.setattr(stage, "execute", lambda *args: None)
    monkeypatch.setattr(stage, "push", lambda *args: None)
    monkeypatch.setattr(stage, "xcom_summary", lambda _: payload)
    monkeypatch.setattr(sys, "argv", ["stage", "src.train", "--workspace", str(tmp_path), "--xcom"])
    stage.main()
    assert json.loads(destination.read_text()) == payload
    assert not (tmp_path / "outside").exists()


def test_failed_gate_preserves_report(tmp_path, monkeypatch):
    import subprocess

    from pipelines import stage

    def failed(module, args, work, env):
        (work / "reports").mkdir()
        (work / "reports/validation.html").write_text("expected gate failure")
        raise subprocess.CalledProcessError(1, ["validate"])

    monkeypatch.setattr(stage, "execute", failed)
    target = tmp_path / "report.html"
    with pytest.raises(subprocess.CalledProcessError):
        stage.run_stage("src.validate_data", outputs={"reports/validation.html": target})
    assert target.read_text() == "expected gate failure"


@pytest.mark.parametrize("args", [["--no-fail=true"], ["--ks-alpha"], ["--no-fail", "--no-fail"]])
def test_invalid_drift_options(args):
    with pytest.raises(ValueError):
        validate_stage_call("src.detect_drift", args)


def test_drift_arguments_unchanged():
    args = ["data/reference.csv", "data/current.csv", "--ks-alpha=0.05", "--no-fail"]
    assert validate_stage_call("src.detect_drift", args) == args
