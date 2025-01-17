"""Test for JanusConfigfile class."""

from __future__ import annotations

from aiida_mlip.data.config import JanusConfigfile


def test_local_file(config_folder):
    """Testing that the local file function works."""
    # Construct a ModelData instance with the local file
    config_path = config_folder / "config_janus_md.yaml"
    config = JanusConfigfile(file=config_path)
    content = config.get_content()
    dictionary = config.as_dictionary
    assert dictionary["ensemble"] == "nvt"
    assert content == config_path.read_text(encoding="utf-8")
    assert isinstance(config, JanusConfigfile)
