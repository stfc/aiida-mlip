"""Some helpers to convert between different formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiida.orm import Bool, Dict, Str, StructureData, TrajectoryData, load_code
from ase.io import read
import numpy as np

from aiida_mlip.helpers.help_load import load_model, load_structure


def convert_numpy(dictionary: dict) -> dict:
    """
    Convert numpy ndarrays in dictionary into lists.

    Parameters
    ----------
    dictionary : dict
        A dictionary with numpy array values to be converted into lists.

    Returns
    -------
    dict
        Converted dictionary.
    """
    new_dict = dictionary.copy()
    for key, value in new_dict.items():
        if isinstance(value, np.ndarray):
            new_dict[key] = value.tolist()
    return new_dict


def xyz_to_aiida_traj(
    traj_file: str | Path,
) -> tuple[StructureData, TrajectoryData]:
    """
    Convert xyz trajectory file to `TrajectoryData` data type.

    Parameters
    ----------
    traj_file : str | Path
        The path to the XYZ file.

    Returns
    -------
    Tuple[StructureData, TrajectoryData]
        A tuple containing the last structure in the trajectory and a `TrajectoryData`
        object containing all structures from the trajectory.
    """
    # Read the XYZ file using ASE
    struct_list = read(traj_file, index=":")

    # Create a TrajectoryData object
    traj = [StructureData(ase=struct) for struct in struct_list]

    return traj[-1], TrajectoryData(traj)


def convert_to_nodes(dictionary: dict, convert_all: bool = False) -> dict:
    """
    Convert each key of the config file to a aiida node.

    Parameters
    ----------
    dictionary : dict
        The dictionary obtained from the config file.
    convert_all : bool
        Define if you want to convert all the parameters or only the main ones.

    Returns
    -------
    dict
        Returns the converted dictionary.
    """
    new_dict = dictionary.copy()
    arch = new_dict["arch"]
    conv = {
        "code": load_code,
        "struct": load_structure,
        "model": lambda v: load_model(v, arch),
        "arch": Str,
        "ensemble": Str,
        "opt_cell_fully": Bool,
    }
    for key, value in new_dict.items():
        if key in conv:
            value = conv[key](value)
        # This is only in the case in which we use the run_from_config function, in that
        # case the config file would be made for aiida specifically not for janus
        elif convert_all:
            if key.endswith("_kwargs") or key.endswith("-kwargs"):
                key = key.replace("-kwargs", "_kwargs")
                value = Dict(value)
            else:
                value = Str(value)
        else:
            continue
        new_dict[key] = value
    return new_dict


def kwarg_to_param(params: dict[str, Any]) -> list[str]:
    """
    Convert a dictionary of kwargs to a set of commandline flags.

    Bools are converted as though ``store_true`` flag keys.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of arguments to convert.

    Returns
    -------
    list[str]
        Commandline arguments as flags.

    Examples
    --------
    >>> kwarg_to_param({"name": "Geoff", "key": True})
    ['--name', 'Geoff', '--key']
    >>> kwarg_to_param({"value": 6, "falsey": False})
    ['--value', '6', '--no-falsey']
    """
    cmdline_params = []

    for key, val in params.items():
        key = key.replace("_", "-")
        match val:
            case bool() if val:
                cmdline_params.append(f"--{key}")
            case bool():
                cmdline_params.append(f"--no-{key}")
            case _:
                cmdline_params.extend((f"--{key}", str(val)))

    return cmdline_params
