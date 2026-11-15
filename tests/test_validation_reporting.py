from types import SimpleNamespace

import pytest

from src.validate_data import observed


@pytest.mark.parametrize("values", [None, []])
def test_no_sample_for_empty_unexpected_values(values):
    result = SimpleNamespace(
        success=False,
        exception_info={},
        result={
            "unexpected_count": 1,
            "unexpected_percent": 10.0,
            "partial_unexpected_list": values,
        },
    )
    assert observed(result) == "1 hors contrat (10.0 %)"


def test_unexpected_sample_is_limited_to_five_values():
    result = SimpleNamespace(
        success=False,
        exception_info={},
        result={
            "unexpected_count": 6,
            "unexpected_percent": 60.0,
            "partial_unexpected_list": list(range(6)),
        },
    )
    assert observed(result) == "6 hors contrat (60.0 %) : 0, 1, 2, 3, 4…"
