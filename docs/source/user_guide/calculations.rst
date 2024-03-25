
==============================
Calculations
==============================

SinglePoint calculation
-----------------------

A `Singlepoint` Calculation represents a `Calcjob` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the `run` or `submit` AiiDA commands.
Below is a usage example with the minimum required parameters. These parameters must be AiiDA data types.


.. code-block:: python


    SinglePointCalculation = CalculationFactory("janus.sp")
    submit(SinglePointCalculation, code=InstalledCode, structure=StructureData)


Refer to the API documentation for additional parameters that can be passed.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_singlepoint.py janus@localhost  --structure "path/to/structure" --model "path/to/model" --precision "float64" --device "cpu"


Geometry Optimisation calculation
---------------------------------

A `GeomOpt` Calculation represents a `Calcjob` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the `run` or `submit` AiiDA commands.
Below is a usage example with some additional geometry optimisation parameters. These parameters must be AiiDA data types.


.. code-block:: python


    GeomOptCalculation = CalculationFactory("janus.opt")
    submit(GeomOptCalculation, code=InstalledCode, structure=StructureData, max_force=Float(0.1), vectors_only=Bool(True))


Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_geomopt.py janus@localhost  --structure "path/to/structure" --model "path/to/model" --precision "float64" --device "cpu"
