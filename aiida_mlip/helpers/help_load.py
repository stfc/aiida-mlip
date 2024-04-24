"""
Helper functions for automatically loading models and strucutres as data nodes.
"""

from pathlib import Path
from typing import Optional, Union

from ase.build import bulk
import ase.io
import click

from aiida.orm import StructureData, load_node

from aiida_mlip.data.model import ModelData


def load_model(
    model: Optional[Union[str, Path]],
    architecture: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> ModelData:
    """
    Load a model from a file path or URL.

    If the string represents a file path, the model will be loaded from that path.
    Otherwise, the model will be downloaded from the specified location.

    Parameters
    ----------
    model : Optional[Union[str, Path]]
        Model file path or a URL for downloading the model.
    architecture : str
        The architecture of the model.
    cache_dir : Optional[Union[str, Path]]
        Directory where to save the dowloaded model.

    Returns
    -------
    ModelData
        The loaded model.
    """
    if model is None:
        loaded_model = None
    elif (file_path := Path(model)).is_file():
        loaded_model = ModelData.local_file(file_path, architecture=architecture)
    else:
        loaded_model = ModelData.download(
            model, architecture=architecture, cache_dir=cache_dir
        )
    return loaded_model


def load_structure(struct: Optional[Union[str, Path, int]] = None) -> StructureData:
    """
    Load a StructureData instance from the given input.

    The input can be either a path to a structure file, a node PK (int),
    or None. If the input is None, a default StructureData instance for NaCl
    with a rocksalt structure will be created.

    Parameters
    ----------
    struct : Optional[Union[str, Path, int]]
        The input value representing either a path to a structure file, a node PK,
        or None.

    Returns
    -------
    StructureData
        The loaded or created StructureData instance.

    Raises
    ------
    click.BadParameter
        If the input is not a valid path to a structure file or a node PK.
    """
    if struct is None:
        structure = StructureData(ase=bulk("NaCl", "rocksalt", 5.63))
    elif isinstance(struct, int) or (isinstance(struct, str) and struct.isdigit()):
        structure_pk = int(struct)
        structure = load_node(structure_pk)
    elif Path.exists(Path(struct)):
        structure = StructureData(ase=ase.io.read(Path(struct)))
    else:
        raise click.BadParameter(
            f"Invalid input: {struct}. Must be either node PK (int) or a valid \
                path to a structure file."
        )
    return structure
