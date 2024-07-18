"""Tests for help_load.py."""

from pathlib import Path

import ase.io
import click
import pytest

from aiida.orm import StructureData

from aiida_mlip.data.model import ModelData
from aiida_mlip.helpers.help_load import load_model, load_structure


def test_load_local_model(model_folder):
    """Test for the load_model function for loading a local file."""
    local_model_path = model_folder / "mace_mp_small.model"
    loaded_model = load_model(local_model_path, architecture="mace_mp")
    assert isinstance(loaded_model, ModelData)


def test_download_model(tmp_path):
    """Test for the load_model function for loading from uri."""
    uri_model = (
        "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model"
    )
    loaded_model = load_model(
        uri_model, architecture="example_architecture", cache_dir=tmp_path
    )
    assert isinstance(loaded_model, ModelData)


def test_load_structure_def():
    """Test for the load_structure function for default structure."""
    default_structure = load_structure(None)
    assert isinstance(default_structure, StructureData)


def test_load_structure_path(structure_folder):
    """Test for the load_structure function for loading from a path."""
    loaded_structure = load_structure(Path(structure_folder / "NaCl.cif"))
    assert isinstance(loaded_structure, StructureData)


def test_load_structure_error():
    """Test for the load_structure function for loading from invalid input."""
    with pytest.raises(click.BadParameter):
        load_structure("non_existent_file.xyz")


def test_load_structure_node(structure_folder):
    """Test for the load_structure function to load structure from node."""
    str_store = StructureData(
        ase=ase.io.read(Path(structure_folder / "NaCl.cif"))
    ).store()
    str_pk = str_store.pk
    loaded_pk = load_structure(str_pk)
    assert isinstance(loaded_pk, StructureData)
