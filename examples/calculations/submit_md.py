"""Example code for submitting a molecular dynamics simulation."""

from __future__ import annotations

import ast

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Str, load_code
from aiida.plugins import CalculationFactory
import click

from aiida_mlip.helpers.help_load import load_model, load_structure


def md(params: dict) -> None:
    """
    Prepare inputs and run a molecular dynamics simulation.

    Parameters
    ----------
    params : dict
        A dictionary containing the input parameters for the calculations

    Returns
    -------
    None
    """
    structure = load_structure(params["struct"])

    # Select model to use
    model = load_model(params["model"], params["arch"])

    # Select calculation to use
    MDCalc = CalculationFactory("mlip.md")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "ensemble": Str(params["ensemble"]),
        "md_kwargs": Dict(params["md_dict"]),
    }

    # Run calculation
    result, node = run_get_node(MDCalc, **inputs)
    print(f"Printing results from calculation: {result}")
    print(f"Printing node of calculation: {node}")


# Arguments and options to give to the cli when running the script
@click.command("cli")
@click.argument("codelabel", type=str)
@click.option(
    "--struct",
    default=None,
    type=str,
    help="Specify the structure (aiida node or path to a structure file)",
)
@click.option(
    "--model",
    default=None,
    type=str,
    help="Specify path or URI of the model to use",
)
@click.option(
    "--arch",
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
    codelabel, struct, model, arch, device, precision, ensemble, md_dict_str
) -> None:
    """Click interface."""
    md_dict = ast.literal_eval(md_dict_str)
    try:
        code = load_code(codelabel)
    except NotExistent as exc:
        print(f"The code '{codelabel}' does not exist.")
        raise SystemExit from exc

    params = {
        "code": code,
        "struct": struct,
        "model": model,
        "arch": arch,
        "device": device,
        "precision": precision,
        "ensemble": ensemble,
        "md_dict": md_dict,
    }

    # Submit MD
    md(params)


if __name__ == "__main__":
    cli()
