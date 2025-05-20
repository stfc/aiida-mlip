Tutorial
========

Running a Geometry optimisation Calculation in aiida-mlip
---------------------------------------------------------

This tutorial guides you through running a single-point calculation using the aiida-mlip package.

Prerequisites
-------------

1. Install ``aiida-mlip``, which also installs `aiida-core <https://github.com/aiidateam/aiida-core>`_ and `janus-core <https://github.com/stfc/janus-core>`_.
2. Set up an AiiDA profile, `computer <https://aiida.readthedocs.io/projects/aiida-core/en/v2.5.1/howto/run_codes.html#how-to-set-up-a-computer>`_ (localhost or remote) and a `code <https://aiida.readthedocs.io/projects/aiida-core/en/v2.5.1/howto/run_codes.html#how-to-create-a-code>`_ (a predefined path for janus-core).

Usage
-----

This code can be run in a `verdi shell` or as a python script through `verdi run` command.

To run a geometry optimisation using aiida-mlip you need to define some inputs as AiiDA data types, to then pass to the calculation.

To start, you will need a structure to optimise. Let's assume the structure is a cif file `/path/to/structure.cif`.
The input structure in aiida-mlip needs to be saved as a StructureData type:

.. code-block:: python

    from aiida.orm import StructureData
    structure = StructureData(ase=read("/path/to/structure.cif"))

Then we need to choose a model and architecture to be used for the calculation and save it as ModelData type, a specific data type of this plugin.
In this example we use MACE with a model that we download from this URI: "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model", and we save the file in the cache folder (default="~/.cache/mlips/"):

.. code-block:: python

    from aiida_mlip.data.model import ModelData
    uri = "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model"
    model = ModelData.from_uri(uri, architecture="mace", cache_dir="/.cache/")

If we already have the model saved in some folder we can save it as:

.. code-block:: python

    model = ModelData.from_local("/path/to/model", architecture="mace")

Another parameter that we need to define as AiiDA type is the code. Assuming the code is saved as `janus` in the `localhost` computer, the code info that are needed can be loaded as follow:

.. code-block:: python

    from aiida.orm import load_code
    code = load_code("janus@localhost")

The other inputs can be set up as AiiDA Str. There is a default for every input except the structure and code. This is a list of possible inputs:

.. code-block:: python

    from aiida.orm import Bool, Float, Str
    inputs = {
        "code": code,
        "model": model,
        "structure": structure,
        "architecture": Str(model.architecture),
        "precision": Str("float64"),
        "device": Str("cpu"),
        "max_force": Float(0.1), # Specific to geometry optimisation: convergence criteria
        "opt_cell_lengths": Bool(False), # Specific to geometry optimisation
        "opt_cell_fully": Bool(True), # Specific to geometry optimisation: to optimise the cell
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

It's worth noting that the architecture is already defined within the model, accessible through the architecture property in the ModelData. Even if not explicitly provided as input, it will be automatically retrieved from the model. The parameters that are not specific to geometry optimisation are the same for the single point calculation.

The calculation must be set:

.. code-block:: python

    from aiida.plugins import CalculationFactory
    geomoptCalculation = CalculationFactory("mlip.opt")

In this case, since we are running a geometry optimisation, the entry point for the calculation is `mlip.opt`. For a single point calculation, the entry point would be `mlip.sp`.

Finally, run the calculation:

.. code-block:: python

    from aiida.engine import run_get_node
    result, node = run_get_node(geomoptCalculation, **inputs)

`results` is a dictionary of the available results obtained from the calculation:

.. code-block:: python

    In : print(result)
    Out :
    {'log_output': <SinglefileData: uuid: 058e153b-f5fb-4799-9686-cc6dcc6f5fbb (pk: 1133)>,
    'xyz_output': <SinglefileData: uuid: 2e8e2f74-39e9-4d3a-a492-02bfa979373b (pk: 1134)>,
    'std_output': <SinglefileData: uuid: a72f2836-1d20-40f6-bcce-d1b56e6b1ba4 (pk: 1135)>,
    'results_dict': <Dict: uuid: 99328f3d-e371-477b-857e-bcbf3353883a (pk: 1136)>,
    'traj_file': <SinglefileData: uuid: 66886994-b856-42f6-abea-af54a8d0eaf8 (pk: 1137)>,
    'traj_output': <TrajectoryData: uuid: b487c8b2-4aca-4c75-b20c-f5d92b625bda (pk: 1138)>,
    'final_structure': <StructureData: uuid: 320b9165-2233-41bc-b14d-b44d8f7f72f3 (pk: 1139)>,
    'remote_folder': <RemoteData: uuid: 4cf9f0cd-20b2-4a47-8dbd-46dbd410a558 (pk: 1131)>,
    'retrieved': <FolderData: uuid: 5601957c-da54-4cd5-9e01-8a215e8ac4cf (pk: 1132)>}


If more information are needed on specifi outputs they can be called like:

.. code-block:: python

    In : result["traj_output"].numsteps
    Out : 3

    In : result["final_structure"].cell #prints cell parameters of the optimised structure
    [[4.0223130461422, -8.6767214011906e-17, 2.7878898106399e-16],
    [2.0111565230711, 3.4834252799327, 2.1832573300987e-16],
    [2.0111565230711, 1.1611417599776, 3.2842048495961]]


Each data type has some properties that can be explored.
In these examples traj_output contains info on the structures at every step of the optimisation(as a TrajectoryData), while final_structure contains info on the optimised structure (as a StructureData).
The properties `numsteps` and `cell` are specific to the respective data types.


while `node` is the node of the calculation

.. code-block:: python

    In : type(node)
    Out : aiida.orm.nodes.process.calculation.calcjob.CalcJobNode

    In : print(node)
    Out: uuid: 1d46ad08-2ea7-4892-9dd6-0240b9aeda8b (pk: 1130) (aiida.calculations:mlip.opt)


The calculation can also be interacted with through verdi cli. Use `verdi process list` to show the list of calculations.

.. code-block:: python

    verdi process list -a
    PK  Created    Process label        Process State     Process status
    ----  ---------  ---------------  ---  ----------------  ----------------------------------
    1130  1m ago    GeomOpt                Finished [0]


.. code-block:: python

    verdi node show 1130
    Property     Value
    -----------  ------------------------------------
    type         GeomOpt
    state        Finished [0]
    pk           1130
    uuid         1d46ad08-2ea7-4892-9dd6-0240b9aeda8b
    label
    description
    ctime        2024-03-19 13:29:58.202562+00:00
    mtime        2024-03-19 13:30:19.461601+00:00
    computer     [2] localhost

    Inputs             PK  Type
    ---------------  ----  -------------
    architecture     1121  Str
    code                2  InstalledCode
    device           1123  Str
    opt_cell_fully   1126  Bool
    log_filename     1128  Str
    max_force        1124  Float
    model            1119  ModelData
    precision        1122  Str
    structure        1120  StructureData
    traj             1129  Str
    opt_cell_lengths 1125  Bool
    xyz_output_name  1127  Str

    Outputs            PK  Type
    ---------------  ----  --------------
    final_structure  1139  StructureData
    log_output       1133  SinglefileData
    remote_folder    1131  RemoteData
    results_dict     1136  Dict
    retrieved        1132  FolderData
    std_output       1135  SinglefileData
    traj_file        1137  SinglefileData
    traj_output      1138  TrajectoryData
    xyz_output       1134  SinglefileData

    Log messages
    ---------------------------------------------
    There are 1 log messages for this calculation
    Run 'verdi process report 1130' to see them

The results can be examined using `verdi calcjob` commands, such as:

.. code-block:: python

    verdi calcjob res 1130
    {
        "cell": [
            [
                4.0223130461422,
                -8.6767214011906e-17,
                2.7878898106399e-16
            ],
            [
                2.0111565230711,
                3.4834252799327,
                2.1832573300987e-16
            ],
            [
                2.0111565230711,
                1.1611417599776,
                3.2842048495961
            ]
        ],
        "forces": [
            [
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0
            ]
        ],
        "info": {
            "energy": -6.7615876501454,
            "free_energy": -6.7615876501454,
            "spacegroup": "P 1",
            "stress": [
                [
                    -0.0001451361211389,
                    -1.122474781947e-17,
                    8.992142627858e-18
                ],
                [
                    -1.122474781947e-17,
                    -0.00014513612113885,
                    -1.0680451921731e-17
                ],
                [
                    8.992142627858e-18,
                    -1.0680451921731e-17,
                    -0.00014513612113892
                ]
            ],
            "unit_cell": "conventional"
        },
        "numbers": [
            11,
            17
        ],
        "pbc": [
            true,
            true,
            true
        ],
        "positions": [
            [
                0.0,
                0.0,
                0.0
            ],
            [
                4.02231305,
                2.32228352,
                1.64210242
            ]
        ],
        "spacegroup_kinds": [
            0,
            1
        ]
    }

`verdi res` contains the results dictionary, which in these calculations is a dictionary containing the `xyz_output` file content.
