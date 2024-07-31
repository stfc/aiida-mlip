"""Workgraph to run high-throughput screening optimisations."""

from pathlib import Path

from aiida_workgraph import WorkGraph, task
from ase.io import read

from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.help_load import load_structure

Geomopt = CalculationFactory("mlip.opt")


@task.graph_builder(outputs=[{"name": "final_structures", "from": "context.relax"}])
def run_opt_calc(folder: Path, janus_opt_inputs: dict) -> WorkGraph:
    """
    Run a geometry optimisation using Geomopt.

    Parameters
    ----------
    folder : Path
        Path to the folder containing input structure files.
    janus_opt_inputs : dict
        Dictionary of inputs for the calculations.

    Returns
    -------
    WorkGraph
        The workgraph containing the optimisation tasks.
    """
    wg = WorkGraph()
    for child in folder.glob("**/*"):
        try:
            read(child.as_posix())
        except Exception:  # pylint: disable=broad-except
            continue
        structure = load_structure(child)
        janus_opt_inputs["struct"] = structure
        relax = wg.add_task(Geomopt, name=f"relax_{child.stem}", **janus_opt_inputs)
        relax.set_context({"final_structure": f"relax.{child.stem}"})
    return wg


def HTSWorkGraph(folder_path: Path, inputs: dict) -> WorkGraph:
    """
    Create and execute a high-throughput workflow for geometry optimisation using MLIPs.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing input structure files.
    inputs : dict
        Dictionary of inputs for the calculations.

    Returns
    -------
    WorkGraph
        The workgraph containing the high-throughput workflow.
    """
    wg = WorkGraph("hts_workflow")

    wg.add_task(
        run_opt_calc, name="opt_task", folder=folder_path, janus_opt_inputs=inputs
    )

    wg.group_outputs = [{"name": "opt_structures", "from": "opt_task.final_structures"}]

    wg.to_html()

    wg.max_number_jobs = 10

    wg.submit()

    return wg
