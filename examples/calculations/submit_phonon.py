"""Example code for submitting phonon calculation."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Float, Str, load_code
from aiida.plugins import CalculationFactory
import click
import yaml

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
        "device": Str(params["device"]),
        "supercell": params["supercell"],
        "nqpoints": params["nqpoints"],
        "fmax": params["fmax"],
        "displacement": Float(params["displacement"]),
    }

    inputs["minimize"] = params["minimize"]
    nohdf5 = params["no_hdf5"]
    inputs["no_hdf5"] = params["no_hdf5"]
    dos = params["dos"]
    inputs["dos"] = params["dos"]
    inputs["pdos"] = params["pdos"]
    pdos = params["pdos"]
    inputs["bands"] = params["bands"]
    bands = params["bands"]
    inputs["symmetrize"] = params["symmetrize"]

    # Only calc_kwargs add if set
    if params["calc_kwargs"]:
        inputs["calc_kwargs"] = Dict(params["calc_kwargs"])

    #############################################################
    #  Run calculation
    #############################################################
    result, node = run_get_node(PhononCalc, **inputs)
    print(f"\n Node of calculation: {node} \n")
    print(
        f"\n use verdi calcjob gotocomputer {node.pk} for a shell in the work directory"
    )

    # start processing the results
    print("\n Results dictionary: \n")
    print(result.keys())

    # print(result["results_dict"].get_dict())
    print(f"\n remote folder {result['remote_folder']} {node.get_remote_workdir()} \n")
    print(f"\n retrieved {result['retrieved']}  \n")
    if not nohdf5:
        print(f"\n force_constant {result['force_constant']} \n")

    if dos:
        print(f"\n density of states {result['dos']} \n")

    if pdos:
        print(f"\n partial density of states {result['pdos']} \n")

    if bands:
        print(f"\n partial density of states {result['band_structure']} \n")

    # dump the dictionary as a yaml file - for inspection / testing
    with open("data.yml", "w") as file:
        yaml.dump(result["results_dict"].get_dict(), file)

    # or dump the phonopy data to a json file
    filepath = "data.json"
    with open(filepath, "w") as file:
        json.dump(result["results_dict"].get_dict(), file, indent=4)

    # access the data such as the supercell matrix
    supercell_matrix = result["results_dict"].get_dict()["supercell_matrix"]
    print(f"\n\n super cell matriz: {supercell_matrix}")

    # verify the hdf5 containing force constants
    if not nohdf5:
        import h5py

        hdf5_path = os.path.join(
            Path(node.get_remote_workdir(), "aiida-force_constants.hdf5")
        )

        with h5py.File(hdf5_path, "r") as f:
            # List all top-level groups/datasets
            print("\n Force constant top-level keys :", list(f.keys()))

    if bands:
        import lzma

        bands_path = os.path.join(
            Path(node.get_remote_workdir(), "aiida-auto_bands.yml.xz")
        )
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
    "--minimize",
    is_flag=True,
    help="minimize the unit cell before calculating the force constants.",
)
@click.option(
    "--nohdf5",
    is_flag=True,
    help="sets flag to true so that force constants are written to yaml file.",
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
    "--fmax", default=0.1, type=float, help="The max force for geometry optimisation."
)
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
    minimize,
    nohdf5,
    dos,
    pdos,
    bands,
    n_qpoints,
    symmetrize,
    fmax,
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
        "fmax": fmax,
        "displacement": displacement,
        "calc_kwargs": calc_kwargs,
    }

    if minimize:
        params["minimize"] = True
    else:
        params["minimize"] = False

    if nohdf5:
        params["no_hdf5"] = True
    else:
        params["no_hdf5"] = False

    if dos:
        params["dos"] = True
    else:
        params["dos"] = False

    if pdos:
        params["pdos"] = True
    else:
        params["pdos"] = False

    if bands:
        params["bands"] = True
    else:
        params["bands"] = False

    if symmetrize:
        params["symmetrize"] = True
    else:
        params["symmetrize"] = False

    if len(qpoint_file) > 0:
        params["qpoint_file"] = qpoint_file

    # Submit single point
    phonon(params)


if __name__ == "__main__":
    cli()

    print("normal exit to the code")
