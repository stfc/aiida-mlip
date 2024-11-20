""" Workgraph to run DFT calculations and use the outputs fpr training a MLIP model."""

from pathlib import Path

from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_workgraph import WorkGraph, task
from ase.io import read
from sklearn.model_selection import train_test_split

from aiida.orm import SinglefileData
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")


@task.graph_builder(outputs=[{"name": "result", "from": "context.pw"}])
def run_pw_calc(folder: Path, dft_inputs: dict) -> WorkGraph:
    """
    Run a quantumespresso calculation using PwRelaxWorkChain.

    Parameters
    ----------
    folder : Path
        Path to the folder containing input structure files.
    dft_inputs : dict
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
        except Exception:  # pylint: disable=broad-except
            continue
        structure = load_structure(child)
        dft_inputs["base"]["structure"] = structure
        dft_inputs["base"]["pw"]["metadata"]["label"] = child.stem
        pw_task = wg.add_task(
            PwRelaxWorkChain, name=f"pw_relax_{child.stem}", **dft_inputs
        )
        pw_task.set_context({"output_structure": f"pw.{child.stem}"})
    return wg


@task.calcfunction()
def create_input(**inputs: dict) -> SinglefileData:
    """
    Create input files from given structures.

    Parameters
    ----------
    **inputs : dict
        Dictionary where keys are names and values are structure data.

    Returns
    -------
    SinglefileData
        A SinglefileData node containing the generated input data.
    """

    input_data = []
    for _, structure in inputs.items():
        ase_structure = structure.to_ase()
        extxyz_str = ase_structure.write(format="extxyz")
        input_data.append(extxyz_str)
    temp_file_path = "tmp.extxyz"
    with open(temp_file_path, "w", encoding="utf8") as temp_file:
        temp_file.write("\n".join(input_data))

    file_data = SinglefileData(file=temp_file_path)

    return file_data


@task.calcfunction(outputs = [{"name": train"},
                              {"name": test"},
                              {"name": "validation"}
                             ])
def split_xyz_file(xyz_file: SinglefileData) -> dict:
    """
    Split an XYZ file into training, testing, and validation datasets.

    Parameters
    ----------
    xyz_file : SinglefileData
        A SinglefileData node containing the XYZ file.

    Returns
    -------
    dict
        A dictionary with keys 'train', 'test', and 'validation', each containing
        SinglefileData nodes for the respective datasets.
    """

    with xyz_file.open() as file:
        lines = file.readlines()

    data = [line.strip() for line in lines if line.strip()]

    train_data, test_validation_data = train_test_split(
        data, test_size=0.4, random_state=42
    )
    test_data, validation_data = train_test_split(
        test_validation_data, test_size=0.5, random_state=42
    )

    train_path = "train.extxyz"
    test_path = "test.extxyz"
    validation_path = "validation.extxyz"

    with open(train_path, "w", encoding="utf8") as f:
        f.write("\n".join(train_data))
    with open(test_path, "w", encoding="utf8") as f:
        f.write("\n".join(test_data))
    with open(validation_path, "w", encoding="utf8") as f:
        f.write("\n".join(validation_data))

    return {
        "train": SinglefileData(file=train_path),
        "test": SinglefileData(file=test_path),
        "validation": SinglefileData(file=validation_path),
    }


@task.calcfunction()
def update_janusconfigfile(janusconfigfile: JanusConfigfile) -> JanusConfigfile:
    """
    Update the JanusConfigfile with new paths for train, test, and validation datasets.

    Parameters
    ----------
    janusconfigfile : JanusConfigfile
        The original JanusConfigfile.

    Returns
    -------
    JanusConfigfile
        A new JanusConfigfile with updated paths.
    """
    print("CHECKPOINT 10")
    janus_dict = janusconfigfile.as_dictionary
    config_parse = janusconfigfile.get_content()

    content = config_parse.replace(janus_dict["train_file"], "train.extxyz")
    content = content.replace(janus_dict["test_file"], "test.extxyz")
    content = content.replace(janus_dict["train_file"], "validation.extxyz")

    new_config_path = "./config.yml"

    with open(new_config_path, "w", encoding="utf8") as file:
        file.write(content)

    return JanusConfigfile(file=new_config_path)


# pylint: disable=unused-variable
def TrainWorkGraph(
    folder_path: Path, inputs: dict, janusconfigfile: JanusConfigfile
) -> WorkGraph:
    """
    Create a workflow for optimising using QE and using the results for training mlips.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing input structure files.
    inputs : dict
        Dictionary of inputs for the calculations.
    janusconfigfile : JanusConfigfile
        File with inputs for janus calculations.

    Returns
    -------
    WorkGraph
        The workgraph containing the training workflow.
    """
    wg = WorkGraph("trainingworkflow")

    pw_task = wg.add_task(
        run_pw_calc, name="pw_relax", folder=folder_path, dft_inputs=inputs
    )

    create_file_task = wg.add_task(create_input, name="create_input")
    wg.add_link(pw_task.outputs["result"], create_file_task.inputs["inputs"])

    split_files_task = wg.add_task(
        split_xyz_file, name="split_xyz", xyz_file=create_file_task.outputs.result
    )

    update_config_task = wg.add_task(
        update_janusconfigfile,
        name="update_janusconfigfile",
        janusconfigfile=janusconfigfile,
    )

    wg.add_link(split_files_task.outputs["_wait"], update_config_task.inputs["_wait"])

    training_calc = CalculationFactory("mlip.train")
    train_inputs = {}
    train_inputs["config_file"] = update_config_task.outputs.result

    train_task = wg.add_task(
        training_calc, name="training", mlip_config=update_config_task.outputs.result
    )
    wg.group_outputs = [{"name": "opt_structures", "from": "pw_task.output_structures"}]
    wg.group_outputs = [{"name": "final_model", "from": "train_task.outputs.model"}]

    wg.to_html()

    wg.max_number_jobs = 10
    wg.submit(wait=True)
    return wg
