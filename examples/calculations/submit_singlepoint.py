"""Example code for submitting single point calculation"""

from pathlib import Path
import sys
from typing import Union

from ase.build import bulk
from ase.io import read
import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Code, Str, StructureData, load_node
from aiida.plugins import CalculationFactory, DataFactory

from aiida_mlip.data.model import ModelData

StructureData = DataFactory("core.structure")


def load_model(string: Union[str, None], architecture: str) -> ModelData:
    """
    Load a model from a given string.

    If the string represents a file path, the model will be loaded from that path.
    Otherwise, the model will be downloaded from the specified location.

    Parameters
    ----------
    string : Union[str, None]
        The string representing either a file path or a URL for downloading the model.
    architecture : str
        The architecture of the model.

    Returns
    -------
    ModelData or None
        The loaded model if successful, otherwise None.
    """
    if string is None:
        model = None
        return model
    file_path = Path(string)
    print(file_path)
    if file_path.is_file():
        model = ModelData.local_file(file_path, architecture=architecture)
        return model
    model = ModelData.download(string, architecture=architecture)
    return model


def load_structure(value: Union[str, int, None]):
    """
    Load a StructureData instance from the given input.

    The input can be either a path to a structure file, a node PK (int),
    or None. If the input is None, a default StructureData instance for NaCl
    with a rocksalt structure will be created.

    Parameters
    ----------
    value : str, int or None
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
    if value is None:
        structure = StructureData(ase=bulk("NaCl", "rocksalt", 5.63))
        return structure
    try:
        # We try to convert the given value to an int, which should be the number of a node
        structure_pk = int(value)
        structure = load_node(structure_pk)
        return structure
    except ValueError as exc:
        # If conversion to int fails, it should be a string
        if Path.exists(value):
            structure = StructureData(ase=read(value))
            return structure
        # If it's none, return error
        raise click.BadParameter(
            f"Invalid input: {value}. Must be either node PK (int) or a valid path to a structure file."
        ) from exc


def singlepoint(params: dict):
    """
    Prepare inputs and run a single point calculation.

    Parameters
    ----------
    params : dict
        A dictionary containing the input parameters for the calculations

    Returns
    -------
    tuple
        A tuple containing the exit code and the node of the calculation.
    """
    structure = load_structure(params["file"])

    # Select model to use
    model = load_model(params["model"], params["architecture"])
    print(model)
    # Select calculation to use
    Singlepointcalc = CalculationFactory("janus.sp")
    print(Singlepointcalc)

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},  # doublecheck this
        "code": params["code"],
        "architecture": Str(params["architecture"]),
        "structure": structure,
        "calctype": Str(params["calctype"]),
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
    }

    # Submit calculation to aiida
    r, s = run_get_node(Singlepointcalc, **inputs)
    print(r)
    print(s)


# Arguments and options to give to the cli when running the script
@click.command("cli")
@click.argument("codelabel", type=str)
@click.option("--calctype", default="singlepoint", type=str)
@click.option(
    "--file",
    default=None,
    type=Union[str, int],
    help="Specify the structure (aiida node or path to a structure file)",
)
@click.option(
    "--model", default=None, type=str, help="Specify path or url of the model to use"
)
@click.option("--architecture", default="mace_mp", type=str)
@click.option("--device", default="cpu", type=str)
@click.option("--precision", default="float64", type=str)
def cli(
    codelabel, calctype, file, model, architecture, device, precision
):  # pylint: disable=too-many-arguments
    """Click interface."""
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print(f"The code '{codelabel}' does not exist.")
        sys.exit(1)

    params = {
        "code": code,
        "calctype": calctype,
        "file": file,
        "model": model,
        "architecture": architecture,
        "device": device,
        "precision": precision,
    }

    # Submit single point
    singlepoint(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
