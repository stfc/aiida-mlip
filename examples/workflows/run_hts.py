"""Example code for submitting single point calculation"""

from ase.build import bulk
import ase.io

from aiida.common import NotExistent
from aiida.engine import WorkChain, run, run_get_node, submit
from aiida.orm import Dict, Int, KpointsData, Str, StructureData, load_code, load_group
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.helpers.help_load import load_structure

HTSWorkChain = WorkflowFactory("mlip.hts")

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# All the other paramenters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
config = JanusConfigfile(
    "/home/federica/aiida-mlip/tests/calculations/configs/config_janus_opt.yaml"
)

# Folder where to get the files
folder = Str("/home/federica/structures_for_test")
# Define calculation to run
entry_point = "mlip.opt"

# Defin inputs for the workchain
inputs = {
    "calc_inputs": {"code": code, "metadata": metadata, "config": config},
    "folder": folder,
    "launch": Str("run_get_node"),
    "group": Int(1),
    "entrypoint": Str("mlip.opt"),
}

result = run(HTSWorkChain, inputs)
print(f"Printing results from calculation: {result}")
