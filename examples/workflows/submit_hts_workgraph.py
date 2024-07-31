"""Example submission for hts workgraph."""

from pathlib import Path

from aiida.orm import load_code

from aiida_mlip.data.model import ModelData
from aiida_mlip.workflows.hts_workgraph import HTSWorkGraph

folder_path = Path("/home/federica/aiida-mlip/tests/workflows/structures/")
inputs = {
    "model": ModelData.from_local(
        "/home/federica/aiida-mlip/tests/data/input_files/mace/mace_mp_small.model",
        architecture="mace_mp",
    ),
    "metadata": {"options": {"resources": {"num_machines": 1}}},
    "code": load_code("janus@localhost"),
}

HTSWorkGraph(folder_path, inputs)
