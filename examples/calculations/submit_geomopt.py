"""Example code for submitting single point calculation"""

from pathlib import Path

import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, Float, Int, Str, load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.help_load import load_model, load_structure


def geomopt(params: dict) -> None:
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
        "opt_kwargs": Dict({"restart": "rest.pkl"}),
        "steps": Int(params["steps"]),
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
@click.option("--steps", default=1000, type=int)
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
    steps,
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
        "steps": steps,
    }

    # Submit single point
    geomopt(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
