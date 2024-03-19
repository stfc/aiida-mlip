"""Example code for submitting single point calculation"""

from pathlib import Path
from typing import Union

from ase.build import bulk
from ase.io import read
import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Float, Str, StructureData, load_code, load_node
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


def load_model(string: Union[str, Path, None], architecture: str) -> ModelData:
    """
    Load a model from a given string.

    If the string represents a file path, the model will be loaded from that path.
    Otherwise, the model will be downloaded from the specified location.

    Parameters
    ----------
    string : Union[str, Path, None]
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
    elif (file_path := Path(string)).is_file():
        model = ModelData.local_file(file_path, architecture=architecture)
    else:
        model = ModelData.download(string, architecture=architecture)
    return model


def load_structure(struct: Union[str, Path, int, None]) -> StructureData:
    """
    Load a StructureData instance from the given input.

    The input can be either a path to a structure file, a node PK (int),
    or None. If the input is None, a default StructureData instance for NaCl
    with a rocksalt structure will be created.

    Parameters
    ----------
    struct : Union[str, Path, int, None]
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
    elif Path.exists(struct):
        structure = StructureData(ase=read(struct))
    else:
        raise click.BadParameter(
            f"Invalid input: {struct}. Must be either node PK (int) or a valid \
                path to a structure file."
        )
    return structure


def singlepoint(params: dict) -> None:
    """
    Prepare inputs and run a single point calculation.

    Parameters
    ----------
    params : dict
        A dictionary containing the input parameters for the calculations

    Returns
    -------
        None
    """
    for key, value in params.items():
        print(key, type(value))

    structure = load_structure(params["file"])

    # Select model to use
    model = load_model(params["model"], params["architecture"])

    # Select calculation to use
    geomoptCalculation = CalculationFactory("janus.opt")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "architecture": Str(params["architecture"]),
        "structure": structure,
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "max_force": Float(params["max_force"]),
        "vectors_only": Bool(params["vectors_only"]),
        "fully_opt": Bool(params["fully_opt"]),
    }

    # Run calculation
    result, node = run_get_node(geomoptCalculation, **inputs)
    print(f"Printing results from calculation: {result}")
    print(f"Printing node of calculation: {node}")


# Arguments and options to give to the cli when running the script
@click.command("cli")
@click.argument("codelabel", type=str)
@click.option(
    "--file",
    default=None,
    type=str,
    help="Specify the structure (aiida node or path to a structure file)",
)
@click.option(
    "--model",
    default=None,
    type=Path,
    help="Specify path or url of the model to use",
)
@click.option("--architecture", default="mace_mp", type=str)
@click.option("--device", default="cpu", type=str)
@click.option("--precision", default="float64", type=str)
@click.option("--max_force", default=0.1, type=float)
@click.option("--vectors_only", default=False, type=bool)
@click.option("--fully_opt", default=False, type=bool)
def cli(
    codelabel,
    file,
    model,
    architecture,
    device,
    precision,
    max_force,
    vectors_only,
    fully_opt,
) -> None:
    # pylint: disable=too-many-arguments
    """Click interface."""
    try:
        code = load_code(codelabel)
    except NotExistent as exc:
        print(f"The code '{codelabel}' does not exist.")
        raise SystemExit from exc

    params = {
        "code": code,
        "file": file,
        "model": model,
        "architecture": architecture,
        "device": device,
        "precision": precision,
        "max_force": max_force,
        "vectors_only": vectors_only,
        "fully_opt": fully_opt,
    }

    # Submit single point
    singlepoint(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
