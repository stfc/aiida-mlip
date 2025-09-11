================================
Training machine learning models
================================

The ``Train`` class represents a ``CalcJob`` object within the AiiDA framework, designed for training machine learning models.

Usage
^^^^^

This calculation can be executed using either the ``run`` or ``submit`` AiiDA commands.
Below is a usage example with some additional training parameters. These parameters must be AiiDA data types.

.. code-block:: python

    TrainCalculation = CalculationFactory("mlip.train")
    submit(TrainCalculation, code=InstalledCode, mlip_config=JanusConfigfile, metadata=Dict({'options': {'output_filename': 'aiida-stdout.txt'}}))


The parameters are provided in a config file. Tha mandatory parameters are:

.. code-block:: yaml

    name: 'test'
    train_file: "./tests/calculations/structures/mlip_train.xyz"
    valid_file: "./tests/calculations/structures/mlip_valid.xyz"
    test_file: "./tests/calculations/structures/mlip_test.xyz"

while the other parameters are optional. Here is an example (can be found in the tests folder) of a config file with more parameters:

.. code-block:: yaml

    name: 'test'
    train_file: "./tests/calculations/structures/mlip_train.xyz"
    valid_file: "./tests/calculations/structures/mlip_valid.xyz"
    test_file: "./tests/calculations/structures/mlip_test.xyz"
    # Optional parameters:
    model: ScaleShiftMACE
    loss: 'universal'
    energy_weight: 1
    forces_weight: 10
    stress_weight: 100
    compute_stress: True
    energy_key: 'dft_energy'
    forces_key: 'dft_forces'
    stress_key: 'dft_stress'
    eval_interval: 2
    error_table: PerAtomRMSE
    # main model params
    interaction_first: "RealAgnosticResidualInteractionBlock"
    interaction: "RealAgnosticResidualInteractionBlock"
    num_interactions: 2
    correlation: 3
    max_ell: 3
    r_max: 4.0
    max_L: 0
    num_channels: 16
    num_radial_basis: 6
    MLP_irreps: '16x0e'
    # end model params
    scaling: 'rms_forces_scaling'
    lr: 0.005
    weight_decay: 1e-8
    ema: True
    ema_decay: 0.995
    scheduler_patience: 5
    batch_size: 4
    valid_batch_size: 4
    max_num_epochs: 1
    patience: 50
    amsgrad: True
    default_dtype: float32
    device: cpu
    distributed: False
    clip_grad: 100
    keep_checkpoints: False
    keep_isolated_atoms: True
    save_cpu: True

It is also possible to fine-tune models using the same type of ``Calcjob``.
In that case some additional parameters must be used: foundation_model and fine_tune.


.. code-block:: python

    inputs = {
         code=InstalledCode,
         mlip_config=JanusConfigfile,
         metadata=Dict({'options': {'output_filename': 'aiida-stdout.txt'}}),
         fine_tune=Bool(True),
         foundation_model=ModelData
         }

    TrainCalculation = CalculationFactory("mlip.train")
    submit(TrainCalculation,inputs)

A model to fine-tune has to be provided as an input, either as a ``ModelData`` type (in which case it has to be a model file), or in the config file at the keyword ``foundation_model``.
If the keyword ``fine_tune`` is True but no model is given either way, it will return an error.

.. note::

    The keyword 'model' and 'foundation_model' refer to two different things.
    'foundation_model' is the path to the model to fine-tune (or a shortcut like 'small', etc).
    'model' refers to the model-type (see `MACE <https://mace-docs.readthedocs.io/en/latest/guide/training.html#model>`_ documentation)


Submission
^^^^^^^^^^

To facilitate the submission process and prepare inputs as AiiDA data types, an example script is provided.
This script can be used as is or by changing, in the file, the path to the config file, then submitted to ``verdi`` as shown

.. code-block:: python

    verdi run submit_train.py
