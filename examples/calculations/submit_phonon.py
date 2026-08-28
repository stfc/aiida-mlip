"""Example code for submitting phonon calculation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Float, Str, load_code
from aiida.plugins import CalculationFactory
import click
import h5py
import yaml

from aiida_mlip.helpers.help_load import load_model, load_structure


def phonon(params: dict[str, any]) -> None:
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
        "device": Str(params["device"]),
        "supercell": params["supercell"],
        "nqpoints": params["nqpoints"],
        "displacement": Float(params["displacement"]),
    }

    for key in "minimize", "no_hdf5", "dos":
        inputs[key] = bool(params[key])
    
    # Only calc_kwargs add if set
    inputs["calc_kwargs"] = Dict(params.get("calc_kwargs", {}))

    #############################################################
    #  Run calculation
    #############################################################
    result, node = run_get_node(PhononCalc, **inputs)
    print(f"Node of calculation: {node} ")
    print(f"use verdi calcjob gotocomputer {node.pk} for a shell in the work directory")

    # start processing the results
    print("Results dictionary: ")
    print(result.keys())

    # print(result["results_dict"].get_dict())
    print(f"remote folder {result['remote_folder']} {node.get_remote_workdir()} ")
    print(f"retrieved {result['retrieved']}  ")
    if not nohdf5:
        print(f"force_constants {result['force_constants']} ")

    if dos:
        print(f"density of states {result['dos']} ")

    if pdos:
        print(f"partial density of states {result['pdos']} ")

    if bands:
        print(f"partial density of states {result['band_structure']} ")

    # dump the dictionary as a yaml file - for inspection / testing
    with open("data.yml", "w") as file:
        yaml.dump(result["results_dict"].get_dict(), file)

    # or dump the phonopy data to a json file
    filepath = "data.json"
    with open(filepath, "w") as file:
        json.dump(result["results_dict"].get_dict(), file, indent=4)

    # access the data such as the supercell matrix
    supercell_matrix = result["results_dict"].get_dict()["supercell_matrix"]
    print(f"supercell matrix: {supercell_matrix}")

    # verify the hdf5 containing force constants
    if not nohdf5:
        hdf5_path = Path(node.get_remote_workdir()) / "aiida-force_constants.hdf5"

        with h5py.File(hdf5_path, "r") as f:
            # List all top-level groups/datasets
            print("Force constant top-level keys :", list(f.keys()))

    if bands:
        import lzma

        bands_path = Path(node.get_remote_workdir()) / "aiida-auto_bands.yml.xz"

        # this will load the data
        with lzma.open(bands_path, "rb") as f:
            data = yaml.safe_load(f)

        # this will write the data out in yaml format
        # warning this could be a (very) big file
        with open("bands_data.yml", "w") as file:
            yaml.dump(data, file)


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
    default="mace",
    type=str,
    help="MLIP architecture to use for calculations.",
)
@click.option(
    "--device", default="cpu", type=str, help="Device to run calculations on."
)
@click.option(
    "--supercell",
    default="2 2 2",
    type=str,
    help="The size of supercell matrix to calculate phonons e.g. 2 2 2.",
)
@click.option(
    "--nohdf5",
    is_flag=True,
    help="write force constants to phonopy yaml, rather than separate HDF5 file.",
)
@click.option("--dos", is_flag=True, help="calculates the density of states.")
@click.option("--pdos", is_flag=True, help="calculates the partial density of states.")
@click.option("--bands", is_flag=True, help="calculates the phonon band structure.")
@click.option(
    "--n-qpoints",
    default="51",
    type=int,
    help="Number of q-points to sample along generated path, including end points.",
)
@click.option("--symmetrize", is_flag=True, help="Symmetrize force constants")
@click.option(
    "--displacement",
    default=0.01,
    type=float,
    help="The displacement employed for numerical derivatives.",
)
@click.option(
    "--qpoint-file", default="", type=str, help="Path to file containing q-point data."
)
@click.option(
    "--calc-kwargs",
    default="{}",
    type=str,
    help="Keyword arguments to pass to calculator.",
)
def cli(
    codelabel,
    struct,
    model,
    arch,
    device,
    supercell,
    nohdf5,
    dos,
    pdos,
    bands,
    n_qpoints,
    symmetrize,
    displacement,
    qpoint_file,
    calc_kwargs,
) -> None:
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
        "supercell": supercell,
        "nqpoints": n_qpoints,
        "displacement": displacement,
        "calc_kwargs": calc_kwargs,
    }

    for param in ("no_hdf5", "dos", "pdos", "symmetrize"):
        if val := getattr(self.inputs, param).value:
            codeinfo.cmdline_params.append("--" + val.replace("_", "-"))
        
    bands = (self.inputs.bands).value
    if bands:
        codeinfo.cmdline_params += [
            "--bands",
        ]
        codeinfo.cmdline_params += ["--n-qpoints", nqpoints]
    
    # Submit phonon calculation
    phonon(params)


if __name__ == "__main__":
    cli()

    print("normal exit to the code")
