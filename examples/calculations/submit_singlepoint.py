"""Example code for submitting single point calculation"""

import sys

from ase.build import bulk
import click

from aiida.common import NotExistent
from aiida.engine import submit
from aiida.orm import Code, Str, StructureData
from aiida.plugins import CalculationFactory, DataFactory

from aiida_mlip.data.model import ModelData

StructureData = DataFactory("structure")


def singlepoint(code):
    """Prepare inputs and run singlepoint calculation"""
    # Example Si structure
    structure = StructureData(ase=bulk("Si", "fcc", 5.43))
    # Structure can be any path to a structure
    # structure = StructureData(ase=ase.io.read(Path("path/to/ciffile.cif")))

    # Select model to use
    model = ModelData.local_file("path/to/model", architecture="mace_mp")

    # Select calculation to use
    Singlepointcalc = CalculationFactory("aiida_mlip.calculations.singlepoint")

    inputs = {
        "structure": structure,
        "model": model,
        "precision": Str("float64"),
        "device": Str("cpu"),
        "code": code,
    }

    submit(Singlepointcalc, **inputs)


# Arguments and options to give to the cli when running the script
@click.command("cli")
@click.argument("codelabel", type=str)
def cli(codelabel):
    """Click interface."""
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print(f"The code '{codelabel}' does not exist.")
        sys.exit(1)

    singlepoint(code)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
