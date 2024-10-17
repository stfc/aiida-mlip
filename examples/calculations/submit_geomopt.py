"""Example code for submitting geometry optimisation calculation"""

import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Float, Int, Str, load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.help_load import load_model, load_structure


def geomopt(params: dict) -> None:
    """
    Prepare inputs and run a geometry optimisation calculation.

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
    geomoptCalculation = CalculationFactory("mlip.opt")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "fmax": Float(params["fmax"]),
        "opt_cell_lengths": Bool(params["opt_cell_lengths"]),
        "opt_cell_fully": Bool(params["opt_cell_fully"]),
        # "opt_kwargs": Dict({"restart": "rest.pkl"}),
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
@click.option("--fmax", default=0.1, type=float, help="Maximum force for convergence.")
@click.option(
    "--opt_cell_lengths",
    default=False,
    type=bool,
    help="Optimise cell vectors, as well as atomic positions.",
)
@click.option(
    "--opt_cell_fully",
    default=False,
    type=bool,
    help="Fully optimise the cell vectors, angles, and atomic positions.",
)
@click.option(
    "--steps", default=1000, type=int, help="Maximum number of optimisation steps."
)
def cli(
    codelabel,
    struct,
    model,
    arch,
    device,
    precision,
    fmax,
    opt_cell_lengths,
    opt_cell_fully,
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
        "struct": struct,
        "model": model,
        "arch": arch,
        "device": device,
        "precision": precision,
        "fmax": fmax,
        "opt_cell_lengths": opt_cell_lengths,
        "opt_cell_fully": opt_cell_fully,
        "steps": steps,
    }

    # Submit single point
    geomopt(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
