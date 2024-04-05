"""Example code for submitting single point calculation"""

import ast

import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Str, load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.help_load import load_model, load_structure


def MD(params: dict) -> None:
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
    MDCalculation = CalculationFactory("janus.md")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "architecture": Str(params["architecture"]),
        "structure": structure,
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "ensemble": Str(params["ensemble"]),
        "md_dict": Dict(params["md_dict"]),
    }

    # Run calculation
    result, node = run_get_node(MDCalculation, **inputs)
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
    type=str,
    help="Specify path or url of the model to use",
)
@click.option(
    "--architecture",
    default="mace_mp",
    type=str,
    help="MLIP architecture to use for calculations.",
)
@click.option(
    "--device", default="cpu", type=str, help="Device to run calculations on."
)
@click.option(
    "--precision", default="float64", type=str, help="Chosen level of precision."
)
@click.option(
    "--ensemble", default="nve", type=str, help="Name of thermodynamic ensemble."
)
@click.option(
    "--md_dict_str",
    default="{}",
    type=str,
    help="String containing a dictionary with other md parameters",
)
def cli(
    codelabel, file, model, architecture, device, precision, ensemble, md_dict_str
) -> None:
    """Click interface."""
    # pylint: disable=too-many-arguments
    md_dict = ast.literal_eval(md_dict_str)
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
        "ensemble": ensemble,
        "md_dict": md_dict,
    }

    # Submit single point
    MD(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
