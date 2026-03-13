"""Example code for submitting phonon calculation."""

from __future__ import annotations

from pathlib import Path
import os
import ast
import yaml
import json

from aiida.common import NotExistent
from aiida.engine import run_get_node
from aiida.orm import Dict, Str, load_code
from aiida.plugins import CalculationFactory
import click

from aiida_mlip.helpers.help_load import load_model, load_structure

from aiida.engine import calcfunction
import aiida.orm as orm
import tempfile
import os.path as path


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
        "supercell": params["supercell"]
    }

    minimize = bool(params["minimize"])
    if minimize:
        inputs["minimize"] = True
    else:
        inputs["minimize"] = False

    nohdf5 = bool(params["no_hdf5"])
    if nohdf5:
        inputs["no_hdf5"] = True
    else:
        inputs["no_hdf5"] = False

    dos = bool(params["dos"])
    if dos:
        inputs["dos"] = True
    else:
        inputs["dos"] = False

    # Only calc_kwargs add if set
    if params["calc_kwargs"]:
        inputs["calc_kwargs"] = Dict(params["calc_kwargs"])

    #############################################################
    #  Run calculation
    #############################################################
    result, node = run_get_node(PhononCalc, **inputs)
    print(f"\n\n Node of calculation: {node} \n\n")
    print(f"\n\n use verdi calcjob gotocomputer {node.pk} to create a shell in the work directory \n\n")

    #start processing the results
    print(f"\n Results dictionary: \n")
    print(result.keys())

    #print(result["results_dict"].get_dict())
    print(f"\n remote folder {result['remote_folder']} {node.get_remote_workdir()} \n")
    print(f"\n retrieved {result['retrieved']}  \n")
    if nohdf5 == False:
        print(f"\n force_constant {result['force_constant']} \n")

    if dos:
        print(f"\n density of states {result['dos']} \n")

    #dump the dictionary as a yaml file - for inspection / testing
    with open('data.yml', 'w') as file:
        yaml.dump(result["results_dict"].get_dict(), file)

    #or dump the phonopy data to a json file
    filepath = "data.json"
    with open(filepath, "w") as file:
        json.dump(result["results_dict"].get_dict(), file, indent=4)  # (The indent is optional, but will make it more human readable)

    #access the data such as the supercell matrix
    supercell_matrix = result['results_dict'].get_dict()['supercell_matrix']
    print(f"\n\n super cell matriz: {supercell_matrix}")

    #verify the hdf5 containing force constants
    if nohdf5 == False:
        import h5py
        hdf5_path = os.path.join(Path(node.get_remote_workdir(), 'aiida-force_constants.hdf5'))

        with h5py.File(hdf5_path, 'r') as f:
            # List all top-level groups/datasets
            print(f"\n Top-level keys :", list(f.keys()))



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
    "--supercell", default="2 2 2", type=str, help="The size of supercell matrix to calculate phonons e.g. 2 2 2."
)
@click.option(
    "--minimize", is_flag=True, help="minimize the unit cell before calculating the force constants."
)
@click.option(
    "--nohdf5", is_flag=True, help="sets flag to true so that force constants are written to yaml file."
)
@click.option(
    "--dos", is_flag=True, help="calculates the density of states."
)
@click.option(
    "--calc-kwargs",
    default="{}",
    type=str,
    help="Keyword arguments to pass to calculator.",
)
def cli(codelabel, struct, model, arch, device, supercell, minimize, nohdf5, dos, calc_kwargs) -> None:
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
    
    # Submit single point
    phonon(params)


if __name__ == "__main__":
    cli()

    print("normal exit to the code")
