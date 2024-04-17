==============================
Calculations
==============================

In these examples, we will assume that the `janus-core <https://github.com/stfc/janus-core>`_ package is installed and saved in the AiiDA database as an `InstalledCode` instance named 'janus@localhost'.

The structure should be a path to a file. Here, the structure file is specified as `path/to/structure`.

.. note::
   Any format that `ASE <https://wiki.fysik.dtu.dk/ase/>`_ can read is a valid structure file for a calculation.

The model file determines the specific MLIP to be used. I can be a local file or a URI to a file to download. In these examples, it is assumed to be a local file located at `path/to/model`.


SinglePoint Calculation
-----------------------

A `Singlepoint` Calculation represents a `Calcjob` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the `run` or `submit` AiiDA commands.
Below is a usage example with the minimum required parameters. These parameters must be AiiDA data types.


.. code-block:: python

    SinglePointCalculation = CalculationFactory("janus.sp")
    submit(SinglePointCalculation, code=InstalledCode, structure=StructureData, metadata="{"options": {"resources": {"num_machines": 1}}}")

The inputs can be grouped into a dictionary:

.. code-block:: python

    inputs = {
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "code": InstalledCode,
            "architecture": Str,
            "structure": StructureData,
            "model": ModelData,
            "precision": Str,
            "device": Str,
        }
    SinglePointCalculation = CalculationFactory("janus.sp")
    submit(SinglePointCalculation, **inputs)


Or they can be passed as a config file. The config file has to be structured as it would be for a janus calculation (refer to janus documentation) and passed as an AiiDA data type itself.
The config file contains the parameters in yaml format:

.. code-block:: yaml

    properties:
      - "energy"
    arch: "mace_mp"
    ensemble: "nvt"
    struct: "path/to/structure.cif"
    model: "path/to/model.model"

And it is used as shown below. Note that some parameters, which are specific to AiiDA, need to be given individually.

.. code-block:: python

    # Add the required inputs for AiiDA
    metadata = {"options": {"resources": {"num_machines": 1}}}
    code = load_code("janus@localhost")

    # All the other parameters are fetched from the config file
    # We want to pass it as an AiiDA data type for provenance
    config = JanusConfigfile("path/to/config.yaml")

    # Define calculation to run
    singlePointCalculation = CalculationFactory("janus.sp")

    # Run calculation
    result, node = run_get_node(
        singlePointCalculation,
        code=code,
        metadata=metadata,
        config=config,
    )

Refer to the API documentation for additional parameters that can be passed.


Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, example scripts are provided.
The submit_singlepoint.py script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_singlepoint.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --precision "float64" --device "cpu"

The submit_using_config.py script can be used to facilitate submission using a config file.

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


.. note::

    As per the singlepoint calculation, the parameters can be provided as a dictionary or config file.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_geomopt.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --precision "float64" --device "cpu"



Molecular Dynamics calculation
------------------------------

An `MD` Calculation represents a `Calcjob` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the `run` or `submit` AiiDA commands.
Below is a usage example with some additional geometry optimisation parameters. These parameters must be AiiDA data types.


.. code-block:: python


    MDCalculation = CalculationFactory("janus.md")
    submit(MDCalculation, code=InstalledCode, structure=StructureData, ensemble=Str("nve") md_dict=Dict({'temp':300,'steps': 4,'traj-every':3,'stats-every':1}))

As per the singlepoint calculation, the parameters can be provided in the form of a dictionary or a config file.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_md.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --ensemble "nve" --md_dict_str "{'temp':300,'steps':4,'traj-every':3,'stats-every':1}"
