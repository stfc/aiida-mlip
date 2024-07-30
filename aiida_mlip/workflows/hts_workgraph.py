""" Workgraph to run DFT calculations and use the outputs fpr training a MLIP model."""

from pathlib import Path

from aiida_mlip.data.model import ModelData
from aiida_workgraph import WorkGraph, task
from sklearn.model_selection import train_test_split
from aiida.orm import Dict, SinglefileData, load_code
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

Geomopt = CalculationFactory("mlip.opt")



@task.graph_builder(outputs=[{"name": "final_structure", "from": "context.pw"}])
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
    for child in folder.glob("**/*xyz"):
        structure = load_structure(child)
        janus_opt_inputs["struct"] = structure
        #janus_opt_inputs['options']['label'] = child.stem
        pw_task = wg.add_task(
            Geomopt, name=f"relax_{child.stem}", **janus_opt_inputs
        )
        pw_task.set_context({"final_structure": f"relax_{child.stem}"})
    return wg


wg = WorkGraph("hts_workflow")
folder_path = Path("/work4/scd/scarf1228/prova_train_workgraph/")
code = load_code("janus_loc@scarf")
inputs = {
    "model" :  ModelData.from_local("/work4/scd/scarf1228/aiida-mlip/tests/calculations/configs/test.model", architecture="mace_mp"),
    "metadata": {"options": {"resources": {"num_machines": 1}}},
    "code":code
}

opt_task = wg.add_task(
    run_pw_calc, name="opt_task", folder=folder_path, janus_opt_inputs=inputs
)
wg.to_html()
print("CHECKPOINT5")
wg.max_number_jobs = 10
wg.submit(wait=True)
