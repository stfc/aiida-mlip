"""Example code for submitting training calculation"""

from pathlib import Path

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# All the other parameters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
mlip_config = JanusConfigfile(
    Path("~/aiida-mlip/tests/calculations/configs/mlip_train.yml")
    .expanduser()
    .resolve()
)

# Define calculation to run
trainCalculation = CalculationFactory("mlip.train")

# Run calculation
result, node = run_get_node(
    trainCalculation,
    code=code,
    metadata=metadata,
    mlip_config=mlip_config,
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
