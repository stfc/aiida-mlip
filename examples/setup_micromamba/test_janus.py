"""Run singlepoint test calculation with janus-core and aiida-mlip."""

from __future__ import annotations

from pathlib import Path
import sys

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Str, StructureData, load_code
from aiida.plugins import CalculationFactory
from ase.build import bulk
import click

from aiida_mlip.data.model import ModelData

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "input_files"
    / "mace"
    / "mace_mp_small.model"
)


def run_test_calculation(
    code,
    arch: str = "mace_mp",
    device: str = "cpu",
    model_path: str | Path | None = None,
) -> None:
    """Run a singlepoint calculation on bulk silicon."""
    atoms = bulk("Si", "diamond", a=5.43)
    structure = StructureData(ase=atoms)

    SinglepointCalc = CalculationFactory("mlip.sp")

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_file = Path(model_path).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = ModelData.from_local(model_file, architecture=arch)

    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": code,
        "struct": structure,
        "arch": Str(arch),
        "model": model,
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
    "--profile",
    default=None,
    help="AiiDA profile to load",
)
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
    "--model",
    "model_path",
    default=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None,
    show_default=True,
    help="Path to MLIP model file",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Device to run on (cpu, cuda)",
)
def cli(
    codelabel: str,
    arch: str,
    device: str,
    model_path: str | None = None,
    profile: str | None = None,
) -> None:
    """CLI interface for testing janus execution."""
    from aiida import load_profile

    load_profile(profile)

    try:
        code = load_code(codelabel)
    except NotExistent as exc:
        print(f"The code '{codelabel}' does not exist.")
        raise SystemExit(1) from exc

    run_test_calculation(code, arch=arch, device=device, model_path=model_path)


if __name__ == "__main__":
    cli()
