"""Tests for help_load.py."""

from pathlib import Path

import click
import pytest

from aiida.orm import StructureData

from aiida_mlip.data.model import ModelData
from aiida_mlip.helpers.help_load import load_model, load_structure


def test_load_model(model_folder, tmp_path):
    """Test for the load_model function."""
    # Test loading a local file
    local_model_path = model_folder / "mace_mp_small.model"
    loaded_model = load_model(local_model_path, architecture="mace_mp")
    assert isinstance(loaded_model, ModelData)

    # Test loading from URL
    url_model = (
        "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model"
    )
    loaded_model = load_model(
        url_model, architecture="example_architecture", cache_dir=tmp_path
    )
    assert isinstance(loaded_model, ModelData)


def test_load_structure(structure_folder):
    """Test for the load_structure function."""
    # Test loading a default structure
    default_structure = load_structure(None)
    assert isinstance(default_structure, StructureData)

    # Test loading from a path
    loaded_structure = load_structure(Path(structure_folder / "NaCl.cif"))
    assert isinstance(loaded_structure, StructureData)

    # Test loading from invalid input
    with pytest.raises(click.BadParameter):
        load_structure("non_existent_file.xyz")
