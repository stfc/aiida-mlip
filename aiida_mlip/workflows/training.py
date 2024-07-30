""" Workgraph to run DFT calculations and use the outputs fpr training a MLIP model."""

from pathlib import Path

from aiida_workgraph import WorkGraph, task
from sklearn.model_selection import train_test_split

from aiida.orm import Dict, SinglefileData, load_code
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
    for child in folder.glob("**/*xyz"):
        structure = load_structure(child)
        dft_inputs["base"]["structure"] = structure
        dft_inputs["base"]["pw"]["metadata"]["label"] = child.stem
        pw_task = wg.add_task(
            PwRelaxWorkChain, name=f"pw_relax_{child.stem}", **dft_inputs
        )
        pw_task.set_context({"result": f"pw_relax_{child.stem}"})
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
    for name, structure in inputs.items():
        ase_structure = structure.to_ase()
        extxyz_str = ase_structure.write(format="extxyz")
        input_data.append(extxyz_str)
    temp_file_path = "tmp.extxyz"
    with open(temp_file_path, "w") as temp_file:
        temp_file.write("\n".join(input_data))

    file_data = SinglefileData(file=temp_file_path)

    return file_data


@task.calcfunction()
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

    with open(train_path, "w") as f:
        f.write("\n".join(train_data))
    with open(test_path, "w") as f:
        f.write("\n".join(test_data))
    with open(validation_path, "w") as f:
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
    janus_dict = janusconfigfile.as_dictionary
    config_parse = janusconfigfile.get_content()

    content = config_parse.replace(janus_dict["train_file"], "train.extxyz")
    content = content.replace(janus_dict["test_file"], "test.extxyz")
    content = content.replace(janus_dict["train_file"], "validation.extxyz")

    new_config_path = "./config.yml"

    with open(new_config_path, "w") as file:
        file.write(content)

    return JanusConfigfile(file=new_config_path)


wg = WorkGraph("trainingworkflow")
folder_path = Path("/work4/scd/scarf1228/prova_train_workgraph/")
code = load_code("qe-7.1@scarf")
inputs = {
    "base": {
        "settings": Dict({"GAMMA_ONLY": True}),
        "pw": {
            "parameters": Dict(
                {
                    "CONTROL": {
                        "calculation": "vc-relax",
                        "nstep": 1200,
                        "etot_conv_thr": 1e-05,
                        "forc_conv_thr": 1e-04,
                    },
                    "SYSTEM": {
                        "ecutwfc": 500,
                        "input_dft": "PBE",
                        "nspin": 1,
                        "occupations": "smearing",
                        "degauss": 0.001,
                        "smearing": "m-p",
                    },
                    "ELECTRONS": {
                        "electron_maxstep": 1000,
                        "scf_must_converge": False,
                        "conv_thr": 1e-08,
                        "mixing_beta": 0.25,
                        "diago_david_ndim": 4,
                        "startingpot": "atomic",
                        "startingwfc": "atomic+random",
                    },
                    "IONS": {
                        "ion_dynamics": "bfgs",
                    },
                    "CELL": {
                        "cell_dynamics": "bfgs",
                        "cell_dofree": "ibrav",
                    },
                }
            ),
            "code": code,
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 4,
                        "num_mpiprocs_per_machine": 32,
                    },
                    "max_wallclock_seconds": 48 * 60 * 60,
                },
            },
        },
    },
    "base_final_scf": {
        "pw": {
            "parameters": Dict(
                {
                    "CONTROL": {
                        "calculation": "scf",
                        "tprnfor": True,
                    },
                    "SYSTEM": {
                        "ecutwfc": 70,
                        "ecutrho": 650,
                        "input_dft": "PBE",
                        "occupations": "smearing",
                        "degauss": 0.001,
                        "smearing": "m-p",
                    },
                    "ELECTRONS": {
                        "conv_thr": 1e-10,
                        "mixing_beta": 0.25,
                        "diago_david_ndim": 4,
                        "startingpot": "atomic",
                        "startingwfc": "atomic+random",
                    },
                }
            ),
            "code": code,
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 32,
                    },
                    "max_wallclock_seconds": 3 * 60 * 60,
                },
            },
        },
    },
}

pw_task = wg.add_task(
    run_pw_calc, name="pw_relax_results", folder=folder_path, dft_inputs=inputs
)

print("CHECKPOINT1")
create_file_task = wg.add_task(create_input, name="create_input")
wg.add_link(pw_task.outputs[0], create_file_task.inputs[0])

print("CHECKPOINT2")
split_files_task = wg.add_task(
    split_xyz_file, name="split_xyz", xyz_file=create_file_task.outputs.result
)
print("CHECKPOINT3")
janusconfigfile_path = "/work4/scd/scarf1228/prova_train_workgraph/mlip_train.yml"
janusconfigfile = JanusConfigfile(file=janusconfigfile_path)
update_config_task = wg.add_task(
    update_janusconfigfile,
    name="update_janusconfigfile",
    janusconfigfile=janusconfigfile,
)

wg.add_link(split_files_task.outputs["result"], update_config_task.inputs["_wait"])
print("CHECKPOINT4")
training_calc = CalculationFactory("mlip.train")
train_inputs = {}
train_inputs["config_file"] = update_config_task.outputs.result
train_task = wg.add_task(
    training_calc, name="training", mlip_config=update_config_task.outputs.result
)

wg.to_html()
print("CHECKPOINT5")
wg.max_number_jobs = 10
wg.submit(wait=True)
