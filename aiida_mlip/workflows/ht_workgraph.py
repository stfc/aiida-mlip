"""Workgraph to run high-throughput calculations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aiida.engine import CalcJob, WorkChain
from aiida.orm import Str
from aiida_workgraph import WorkGraph, task
from ase.io import read

from aiida_mlip.helpers.help_load import load_structure


@task.graph_builder(outputs=[{"name": "final_structures", "from": "context.structs"}])
def build_ht_calc(
    calc: CalcJob | Callable | WorkChain | WorkGraph,
    folder: Path | str | Str,
    calc_inputs: dict,
    input_struct_key: str = "struct",
    final_struct_key: str = "final_structure",
    recursive: bool = True,
) -> WorkGraph:
    """
    Build high throughput calculation WorkGraph.

    The `calc` must take a structure, by default `struct`, as one of its inputs.
    Tasks will then be created to carry out the calculation for each structure file in
    `folder`.

    Parameters
    ----------
    calc : Union[CalcJob, Callable, WorkChain, WorkGraph]
        Calculation to be performed on all structures.
    folder : Union[Path, str, Str]
        Path to the folder containing input structure files.
    calc_inputs : dict
        Dictionary of inputs, shared by all the calculations. Must not contain
        `struct_key`.
    input_struct_key : str
        Keyword for input structure for `calc`. Default is "struct".
    final_struct_key : str
        Key for final structure output from `calc`. Default is "final_structure".
    recursive : bool
        Whether to search `folder` recursively. Default is True.

    Returns
    -------
    WorkGraph
        The workgraph with calculation tasks for each structure.

    Raises
    ------
    FileNotFoundError
        If `folder` has no valid structure files.
    """
    wg = WorkGraph()
    structure = None

    if isinstance(folder, Str):
        folder = Path(folder.value)
    if isinstance(folder, str):
        folder = Path(folder)

    pattern = "**/*" if recursive else "*"
    for file in filter(Path.is_file, folder.glob(pattern)):
        try:
            read(file)
        except Exception:
            continue
        structure = load_structure(file)
        calc_inputs[input_struct_key] = structure
        calc_task = wg.add_task(
            calc,
            name=f"calc_{file.stem}",
            **calc_inputs,
        )
        print(calc_task)
        print(f"calc_{file.stem}")
        print({f"structs.{file.stem}": final_struct_key})
        wg.update_ctx({f"structs.{file.stem}": final_struct_key})
        print(wg.tasks)

    # wg.outputs.result_H2O = wg.tasks.calc_H2O
    print(wg.tasks.calc_H2O.outputs)

    if structure is None:
        raise FileNotFoundError(
            f"{folder} is empty or has no readable structure files."
        )

    return wg


def get_ht_workgraph(
    calc: CalcJob | Callable | WorkChain | WorkGraph,
    folder: Path | str | Str,
    calc_inputs: dict,
    input_struct_key: str = "struct",
    final_struct_key: str = "final_structure",
    recursive: bool = True,
    max_number_jobs: int = 10,
) -> WorkGraph:
    """
    Get WorkGraph to carry out calculation on all structures in a directory.

    Parameters
    ----------
    calc : Union[CalcJob, Callable, WorkChain, WorkGraph]
        Calculation to be performed on all structures.
    folder : Union[Path, str, Str]
        Path to the folder containing input structure files.
    calc_inputs : dict
        Dictionary of inputs, shared by all the calculations. Must not contain
        `struct_key`.
    input_struct_key : str
        Keyword for input structure for `calc`. Default is "struct".
    final_struct_key : str
        Key for final structure output from `calc`. Default is "final_structure".
    recursive : bool
        Whether to search `folder` recursively. Default is True.
    max_number_jobs : int
        Max number of subprocesses running within the WorkGraph. Default is 10.

    Returns
    -------
    WorkGraph
        The workgraph ready to be submitted.
    """
    wg = WorkGraph("ht_calculation")

    wg.add_task(
        build_ht_calc,
        name="ht_calc",
        calc=calc,
        folder=folder,
        calc_inputs=calc_inputs,
        input_struct_key=input_struct_key,
        final_struct_key=final_struct_key,
        recursive=recursive,
    )

    # wg.group_outputs = [
    #     {"name": "final_structures", "from": "ht_calc.final_structures"}
    # ]
    wg.max_number_jobs = max_number_jobs

    return wg
