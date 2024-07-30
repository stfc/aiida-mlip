""" Workgraph to run DFT calculations and use the outputs fpr training a MLIP model."""

from pathlib import Path

from aiida_mlip.data.model import ModelData
from aiida_workgraph import WorkGraph, task
from sklearn.model_selection import train_test_split
from aiida.orm import Dict, SinglefileData, load_code
from aiida.plugins import CalculationFactory, WorkflowFactory
from ase.io import read
from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

Geomopt = CalculationFactory("mlip.opt")

@task.graph_builder(outputs=[{"name": "final_structures", "from": "context.relax"}])
def run_pw_calc(folder: Path, janus_opt_inputs: dict) -> WorkGraph:
    """
    Run a quantumespresso calculation using PwRelaxWorkChain.

    Parameters
    ----------
    folder : Path
        Path to the folder containing input structure files.
    janus_opt_inputs : dict
        Dictionary of inputs for the DFT calculations.

    Returns
    -------
    WorkGraph
        The work graph containing the PW relaxation tasks.
    """
    wg = WorkGraph()
    for child in folder.glob("**/*"):
        try:
            read(child.as_posix())
        except Exception:
            continue
        structure = load_structure(child)
        janus_opt_inputs["struct"] = structure
        relax = wg.add_task(
            Geomopt, name=f"relax_{child.stem}", **janus_opt_inputs
        )    
        relax.set_context({"final_structure": f"relax.{child.stem}"})
    return wg

def HTSWorkGraph(folder_path, inputs):
    wg = WorkGraph("hts_workflow")

    opt_task = wg.add_task(
        run_pw_calc, name="opt_task", folder=folder_path, janus_opt_inputs=inputs
    )

    wg.group_outputs = [{"name": "opt_structures", "from": "opt_task.final_structures"}]


    wg.to_html()


    wg.max_number_jobs = 10

    wg.submit(wait=True)


