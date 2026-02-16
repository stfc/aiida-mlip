"""Example code for submitting phonon calculation."""

from __future__ import annotations

import ast

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Str, load_code
from aiida.plugins import CalculationFactory
import click

from aiida_mlip.helpers.help_load import load_model, load_structure


def phonon(params: dict) -> None:
    """
    Prepare inputs and run a phonon calculation.

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
    PhononCalc = CalculationFactory("mlip.ph")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "model": model,
        "device": 'cpu' #Str(params["device"]),
    }

    # Only calc_kwargs add if set
    if params["calc_kwargs"]:
        inputs["calc_kwargs"] = Dict(params["calc_kwargs"])

    # Run calculation
    result, node = run_get_node(PhononCalc, **inputs)
    print(f"\n\n Printing node of calculation: {node} \n\n")
    print(f"Printing results from calculation: {result}")

    print(f"Results dictionary: \n\n")
    print(result.keys())

    print(result["results_dict"].get_dict())

    print(f"\n\n\n Energy: {result['results_dict'].get_dict()['crystal']['cell_vectors']}")

    remote = load_node(result["remote_folder"].pk)
    retrieved = load_node(result["retrieved"].pk)

#   print("remote",remote)
#   print("retrieved", retrieved)

#   print("outputs", node.outputs)


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
    "--calc-kwargs",
    default="{}",
    type=str,
    help="Keyword arguments to pass to calculator.",
)
def cli(codelabel, struct, model, arch, device, calc_kwargs) -> None:
    """Click interface."""
    calc_kwargs = ast.literal_eval(calc_kwargs)

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
        "calc_kwargs": calc_kwargs,
    }

    # Submit single point
    phonon(params)


if __name__ == "__main__":
    cli()

    print("normal exit to the code")
