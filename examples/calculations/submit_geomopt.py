"""Example code for submitting geometry optimisation calculation"""

import click

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, Float, Int, Str, load_code
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
    geomoptCalculation = CalculationFactory("janus.opt")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "fmax": Float(params["fmax"]),
        "vectors_only": Bool(params["vectors_only"]),
        "fully_opt": Bool(params["fully_opt"]),
        "opt_kwargs": Dict({"restart": "rest.pkl"}),
        "steps": Int(params["steps"]),
    }
    if model is not None:
        inputs["model"] = model

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
    help="Specify path or url of the model to use",
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
    "--vectors_only",
    default=False,
    type=bool,
    help="Optimise cell vectors, as well as atomic positions.",
)
@click.option(
    "--fully_opt",
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
        "struct": struct,
        "model": model,
        "arch": arch,
        "device": device,
        "precision": precision,
        "fmax": fmax,
        "vectors_only": vectors_only,
        "fully_opt": fully_opt,
        "steps": steps,
    }

    # Submit single point
    geomopt(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
