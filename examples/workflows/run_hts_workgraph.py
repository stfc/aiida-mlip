from aiida_mlip.workflows.hts_workgraph import HTSWorkGraph
from pathlib import Path
from aiida_mlip.data.model import ModelData
from aiida.orm import load_code

folder_path = Path("/work4/scd/scarf1228/prova_train_workgraph/")
inputs = {
    "model" :  ModelData.from_local("/work4/scd/scarf1228/aiida-mlip/tests/calculations/configs/test.model", architecture="mace_mp"),
    "metadata": {"options": {"resources": {"num_machines": 1}}},
    "code": load_code("janus_loc@scarf")
}

HTSWorkGraph(folder_path, inputs)
