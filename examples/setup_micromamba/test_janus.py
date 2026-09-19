"""Run singlepoint test calculation with janus-core and aiida-mlip."""

from __future__ import annotations

import sys

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Str, StructureData, load_code
from aiida.plugins import CalculationFactory
from ase.build import bulk
import click


def run_test_calculation(code, arch: str = "mace_mp", device: str = "cpu") -> None:
    """Run a singlepoint calculation on bulk silicon."""
    atoms = bulk("Si", "diamond", a=5.43)
    structure = StructureData(ase=atoms)

    SinglepointCalc = CalculationFactory("mlip.sp")

    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": code,
        "struct": structure,
        "arch": Str(arch),
        "device": Str(device),
    }

    print(f"Submitting singlepoint calculation using code: {code.full_label}...")
    results, node = run_get_node(SinglepointCalc, **inputs)
    print(f"Calculation finished. PK: {node.pk}, Exit status: {node.exit_status}")
    print(f"Results: {results}")

    if not node.is_finished_ok:
        print(f"ERROR: Calculation failed with exit status {node.exit_status}")
        sys.exit(1)


@click.command("cli")
@click.option(
    "--codelabel",
    default="janus@localhost",
    show_default=True,
    help="Code label in AiiDA",
)
@click.option(
    "--arch",
    default="mace_mp",
    show_default=True,
    help="MLIP architecture",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device to run on (cpu, cuda)",
)
def cli(codelabel: str, arch: str, device: str) -> None:
    """CLI interface for testing janus execution."""
    try:
        code = load_code(codelabel)
    except NotExistent as exc:
        print(f"The code '{codelabel}' does not exist.")
        raise SystemExit(1) from exc

    run_test_calculation(code, arch=arch, device=device)


if __name__ == "__main__":
    cli()
