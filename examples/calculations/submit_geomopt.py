"""Example code for submitting geometry optimisation calculation."""

from __future__ import annotations

import ast

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, Float, Int, Str, load_code
from aiida.plugins import CalculationFactory
import click

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
    GeomoptCalc = CalculationFactory("mlip.opt")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "model": model,
        "device": Str(params["device"]),
        "fmax": Float(params["fmax"]),
        "opt_cell_lengths": Bool(params["opt_cell_lengths"]),
        "opt_cell_fully": Bool(params["opt_cell_fully"]),
        "steps": Int(params["steps"]),
    }

    # Only calc_kwargs add if set
    if params["calc_kwargs"]:
        inputs["calc_kwargs"] = Dict(params["calc_kwargs"])

    # Only minimize_kwargs add if set
    if params["minimize_kwargs"]:
        inputs["minimize_kwargs"] = Dict(params["minimize_kwargs"])

    # Run calculation
    result, node = run_get_node(GeomoptCalc, **inputs)
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
    "--calc-kwargs",
    default="{}",
    type=str,
    help="Keyword arguments to pass to calculator.",
)
@click.option("--fmax", default=0.1, type=float, help="Maximum force for convergence.")
@click.option(
    "--opt-cell-lengths",
    default=False,
    type=bool,
    help="Optimise cell vectors, as well as atomic positions.",
)
@click.option(
    "--opt-cell-fully",
    default=False,
    type=bool,
    help="Fully optimise the cell vectors, angles, and atomic positions.",
)
@click.option(
    "--steps", default=1000, type=int, help="Maximum number of optimisation steps."
)
@click.option(
    "--minimize-kwargs",
    default="{}",
    type=str,
    help=(
        "Keyword arguments to pass to geometry optimizer, including 'opt_kwargs', "
        "'filter_kwargs', and 'traj_kwargs'."
    ),
)
def cli(
    codelabel,
    struct,
    model,
    arch,
    device,
    calc_kwargs,
    fmax,
    opt_cell_lengths,
    opt_cell_fully,
    steps,
    minimize_kwargs,
) -> None:
    """Click interface."""
    calc_kwargs = ast.literal_eval(calc_kwargs)
    minimize_kwargs = ast.literal_eval(minimize_kwargs)

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
        "fmax": fmax,
        "opt_cell_lengths": opt_cell_lengths,
        "opt_cell_fully": opt_cell_fully,
        "steps": steps,
        "minimize_kwargs": minimize_kwargs,
    }

    # Submit single point
    geomopt(params)


if __name__ == "__main__":
    cli()
