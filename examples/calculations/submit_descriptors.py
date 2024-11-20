"""Example code for submitting descriptors calculation."""

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Bool, Str, load_code
from aiida.plugins import CalculationFactory
import click

from aiida_mlip.helpers.help_load import load_model, load_structure


def descriptors(params: dict) -> None:
    """
    Prepare inputs and run a descriptors calculation.

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
    DescriptorsCalc = CalculationFactory("mlip.descriptors")

    # Define inputs
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": params["code"],
        "arch": Str(params["arch"]),
        "struct": structure,
        "model": model,
        "precision": Str(params["precision"]),
        "device": Str(params["device"]),
        "invariants_only": Bool(params["invariants_only"]),
        "calc_per_element": Bool(params["calc_per_element"]),
        "calc_per_atom": Bool(params["calc_per_atom"]),
    }

    # Run calculation
    result, node = run_get_node(DescriptorsCalc, **inputs)
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
    "--invariants-only",
    default=False,
    type=bool,
    help="Only calculate invariant descriptors.",
)
@click.option(
    "--calc-per-element",
    default=False,
    type=bool,
    help="Calculate mean descriptors for each element.",
)
@click.option(
    "--calc-per-atom",
    default=False,
    type=bool,
    help="Calculate descriptors for each atom.",
)
def cli(
    codelabel,
    struct,
    model,
    arch,
    device,
    precision,
    invariants_only,
    calc_per_element,
    calc_per_atom,
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
        "invariants_only": invariants_only,
        "calc_per_element": calc_per_element,
        "calc_per_atom": calc_per_atom,
    }

    # Submit descriptors
    descriptors(params)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
