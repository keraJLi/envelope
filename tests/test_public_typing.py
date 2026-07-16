"""Runtime-facing expectations for Envelope's inline typing surface."""

from importlib import resources

import pytest

import envelope
import envelope.typing as envelope_typing


@pytest.mark.parametrize("name", ["Array", "Info", "Key", "PyTree", "State"])
def test_typing_names_are_exported_from_package_root(name):
    assert name in envelope.__all__
    assert getattr(envelope, name) is getattr(envelope_typing, name)


def test_package_contains_pep561_marker_for_built_wheels():
    assert resources.files("envelope").joinpath("py.typed").is_file()
