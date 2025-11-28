============
Calculations
============

In these examples, we will assume that the `janus-core <https://github.com/stfc/janus-core>`_ package is installed and saved in the AiiDA database as an ``InstalledCode`` instance named 'janus@localhost'.

The structure should be a path to a file. Here, the structure file is specified as ``path/to/structure``.

.. note::
   Any format that `ASE <https://ase-lib.org>`_ can read is a valid structure file for a calculation.

The model file determines the specific MLIP to be used. It can be a local file or a URI to a file to download. In these examples, it is assumed to be a local file located at ``path/to/model``.


SinglePoint Calculation
-----------------------

A ``Singlepoint`` Calculation represents a ``Calcjob`` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the ``run`` or ``submit`` AiiDA commands.
Below is a usage example with the minimum required parameters. These parameters must be AiiDA data types.


.. code-block:: python

    SinglePointCalculation = CalculationFactory("mlip.sp")
    submit(SinglePointCalculation, code=InstalledCode, structure=StructureData, metadata={"options": {"resources": {"num_machines": 1}}})

The inputs can be grouped into a dictionary:

.. code-block:: python

    inputs = {
            "metadata": {"options": {"resources": {"num_machines": 1}}},
            "code": InstalledCode,
            "architecture": Str,
            "structure": StructureData,
            "model": ModelData,
            "device": Str,
            "calc_kwargs": Dict,
        }
    SinglePointCalculation = CalculationFactory("mlip.sp")
    submit(SinglePointCalculation, **inputs)


Or they can be passed as a config file. The config file has to be structured as it would be for a janus calculation (refer to `janus documentation <https://stfc.github.io/janus-core/>`_ ) and passed as an AiiDA data type itself.
The config file contains the parameters in yaml format:

.. code-block:: yaml

    properties:
      - "energy"
    arch: "mace_mp"
    device: "cpu"
    struct: "path/to/structure.cif"
    model: "path/to/model.model"
    calc_kwargs:
      dispersion: True


And it is used as shown below. Note that some parameters, which are specific to AiiDA, need to be given individually.

.. code-block:: python

    # Add the required inputs for AiiDA
    metadata = {"options": {"resources": {"num_machines": 1}}}
    code = load_code("janus@localhost")

    # All the other parameters are fetched from the config file
    # We want to pass it as an AiiDA data type for provenance
    config = JanusConfigfile("path/to/config.yaml")

    # Define calculation to run
    SinglePointCalculation = CalculationFactory("mlip.sp")

    # Run calculation
    result, node = run_get_node(
        SinglePointCalculation,
        code=code,
        metadata=metadata,
        config=config,
    )

If a parameter is defined twice, in the config file and manually, the manually defined one will overwrite the config one.
If for example the same config file as before is used, but this time the parameter ``struct`` is added to the launch function, the code would look like this:

.. code-block:: python

    # Run calculation
    result, node = run_get_node(
        SinglePointCalculation,
        code=code,
        struct=StructureData(ase=read("path/to/structure2.xyz"))
        metadata=metadata,
        config=config,
    )

In this case  the structure used is going to be ``path/to/structure2.xyz`` rather than ``path/to/structure.cif``, which was defined in the config file.

Refer to the API documentation for additional parameters that can be passed.
Some parameters are not required and don't have a default value set in ``aiida-mlip``. In that case the default values will be the same as `janus-core <https://github.com/stfc/janus-core>`_
The only default parameters defined in ``aiida-mlip`` are the names of the input and output files, as they do not affect the results of the calculation itself, and are needed in AiiDA to parse the results.


Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, example scripts are provided.
The submit_singlepoint.py script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. note::

    The example files are set up with default values, ensuring that calculations runs even if no input is provided via the cli.
    However, the ``aiida-mlip`` code itself does require certain parameters, (e.g. the structure on which to perform the calculation).


.. code-block:: python

    verdi run submit_singlepoint.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --device "cpu"

The ``submit_using_config.py`` script provides an example of submission using a config file.

.. note::

    The structure and model are hard-coded into ``submit_using_config.py`` to avoid
    issues with relative paths. These should be modified, or removed and set through
    the configuration file, for your structure and model of interest.


Geometry Optimisation calculation
---------------------------------

A ``GeomOpt`` Calculation represents a ``Calcjob`` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the ``run`` or ``submit`` AiiDA commands.
Below is a usage example with some additional geometry optimisation parameters. These parameters must be AiiDA data types.


.. code-block:: python


    GeomOptCalculation = CalculationFactory("mlip.opt")
    submit(GeomOptCalculation, code=InstalledCode, structure=StructureData, max_force=Float(0.1), opt_cell_lengths=Bool(True))


.. note::

    As per the singlepoint calculation, the parameters can be provided as a dictionary or config file.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_geomopt.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --device "cpu"



Molecular Dynamics calculation
------------------------------

An ``MD`` Calculation represents a ``Calcjob`` object within the AiiDA framework.


Usage
^^^^^

This calculation can be executed using either the ``run`` or ``submit`` AiiDA commands.
Below is a usage example with some additional geometry optimisation parameters. These parameters must be AiiDA data types.


.. code-block:: python


    MDCalculation = CalculationFactory("mlip.md")
    submit(MDCalculation, code=InstalledCode, structure=StructureData, ensemble=Str("nve"), md_dict=Dict({'temp':300,'steps': 4,'traj-every':3,'stats-every':1}))

.. note::

   As per the singlepoint calculation, the parameters can be provided as a dictionary or config file.

Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is, submitted to verdi, and the parameters passed as strings to the CLI.
They will be converted to AiiDA data types by the script itself.

.. code-block:: python

    verdi run submit_md.py "janus@localhost" --structure "path/to/structure" --model "path/to/model" --ensemble "nve" --md-dict-str "{'temp':300,'steps':4,'traj-every':3,'stats-every':1}"
