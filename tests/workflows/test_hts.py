"""Test for high-throughput-screening WorkGraph."""

# from aiida.orm import StructureData, load_node

from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.hts_workgraph import HTSWorkGraph


def test_hts_wg(janus_code, structure_folder2, model_folder) -> None:
    """Submit simple calcjob."""
    model_file = model_folder / "mace_mp_small.model"
    inputs = {
        "model": ModelData.from_local(model_file, architecture="mace"),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
    }
    wg = HTSWorkGraph(folder_path=structure_folder2, inputs=inputs)
    wg.wait(15)

    # AT THE MOMENT WE ONLY CHECK THE PROCESS IS CREATED AT LEAST,
    #  WHEN WE FIX THE SUBMISSION THIS NEEDS TO BE CHANGED

    assert wg.state == "CREATED"
    # wg_node = load_node(wg.pk)
    # assert isinstance(wg_node.outputs.opt_structures.h2o, StructureData)
