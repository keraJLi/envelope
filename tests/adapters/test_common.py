import warnings

import pytest

from envelope.adapters._common import _capture_horizon, warn_if_wrapper_overlap


def test_warn_if_wrapper_overlap_reports_only_overlapping_arguments():
    supplied_args = {
        "safe_backend_option": True,
        "episode_length": 17,
        "auto_reset": False,
    }

    with pytest.warns(
        UserWarning,
        match=(
            r"Explicit Example backend settings may overlap with Envelope wrappers: "
            r"auto_reset, episode_length\."
        ),
    ):
        warn_if_wrapper_overlap(
            "Example",
            supplied_args,
            ("episode_length", "auto_reset", "batch_size"),
        )


def test_warn_if_wrapper_overlap_ignores_unrelated_arguments():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_wrapper_overlap(
            "Example", {"safe_backend_option": True}, ("auto_reset",)
        )


def test_warn_if_wrapper_overlap_handles_named_optional_arguments():
    with pytest.warns(UserWarning, match=r"env_params\."):
        warn_if_wrapper_overlap("Example", {}, ("env_params",), env_params=object())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_wrapper_overlap("Example", {}, ("env_params",), env_params=None)


def test_warn_if_wrapper_overlap_accepts_none_supplied_args():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_wrapper_overlap("Example", None, ("auto_reset",))


def test_capture_horizon_checks_finiteness_before_conversion():
    assert _capture_horizon(17.9) == 17
    assert _capture_horizon(None) is None
    assert _capture_horizon(float("inf")) is None
    assert _capture_horizon(float("-inf")) is None
