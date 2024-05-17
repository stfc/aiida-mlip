"""Example code for submitting training calculation"""

from pathlib import Path

from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.config import JanusConfigfile

# Add the required inputs for aiida
metadata = {"options": {"resources": {"num_machines": 1}}}
code = load_code("janus@localhost")

# All the other paramenters we want them from the config file
# We want to pass it as a AiiDA data type for the provenance
mlip_config = JanusConfigfile(
    (
        Path(__file__).parent / "../../tests/calculations/configs/mlip_train.yml"
    ).resolve()
)

# Define calculation to run
trainCalculation = CalculationFactory("janus.train")

# Run calculation
result, node = run_get_node(
    trainCalculation,
    code=code,
    metadata=metadata,
    mlip_config=mlip_config,
)
print(f"Printing results from calculation: {result}")
print(f"Printing node of calculation: {node}")
