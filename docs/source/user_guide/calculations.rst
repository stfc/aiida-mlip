
==============================
Calculations
==============================

SinglePoint calculation
-----------------------

A Single Point Calculation represents a `Calcjob` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the `run` or `submit` AiiDA commands.
Below is a usage example with the minimum required parameters. These parameters must be AiiDA data types.


.. code-block:: python

    submit(SinglePointCalculation, code=AbstractCode, calctype=Str, model=ModelData, structure=StructureData)


Refer to the API documentation for additional parameters that can be passed.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_singlepoint.py janus@localhost --calctype "singlepoint" --structure "path/to/structure" --model "path/to/model" --precision "float64" --device "cpu"
