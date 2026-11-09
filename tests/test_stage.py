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
