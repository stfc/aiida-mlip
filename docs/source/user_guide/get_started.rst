===============
Getting started
===============

Installation
++++++++++++

We suggest creating a new `virtual environment <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_ and activating it before running the commands below.

The latest stable release of ``aiida-mlip``, including its dependencies, can be installed from PyPI by running:

.. code-block:: bash

    python3 -m pip install aiida-mlip

To get all the latest changes, ``aiida-mlip`` can also be installed from GitHub:

.. code-block:: bash

    python3 -m pip install git+https://github.com/stfc/aiida-mlip.git

By default, no machine learnt interatomic potentials (MLIPs) will be installed with ``aiida-mlip``.
However, ``aiida-mlip`` currently provides an ``extra``, allowing MACE to be installed:

.. code-block:: bash

   python3 -m pip install aiida-mlip[mace]

For additional MLIPs, it is recommended that the ``extra`` dependencies provided by ``janus-core`` are used.
For example, to install CHGNet and SevenNet, run:

.. code-block:: bash

    python3 -m pip install janus-core[chgnet,sevennet]

Please refer to the ``janus-core`` `documentation <https://stfc.github.io/janus-core/getting_started/getting_started.html#installation>`_ for further details.

Once ``aiida-mlip`` and the desired MLIP calculators are installed, run::

    verdi presto  # better to set up a new profile
    verdi plugin list aiida.calculations  # should now show your calculation plugins

Then, use ``verdi code setup`` with the ``janus`` input plugin
to set up an AiiDA code for aiida-mlip. The `aiida docs <https://aiida.readthedocs.io/projects/aiida-core/en/stable/howto/run_codes.html#how-to-create-a-code>`_ go over how to create a code.



.. note::
    Configuring a message broker like RabbitMQ is optional, but highly recommended to avoid errors and enable `full functionality <https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html#quick-install-limitations>`_ of AiiDA.
    If you have not set up RabbitMQ, you will still be able to ``run`` processes (as shown in the `tutorial notebooks <https://github.com/stfc/aiida-mlip/tree/main/examples/tutorials>`_) but not be able to ``submit`` them.
    If a broker is detected, the ``verdi presto`` command can automatically configure a presto profile, including the computer, database, and broker.
    You’ll also need to set up a code for ``janus-core`` so it can be recognised by AiiDA. Note that PostgreSQL is not configured by default.
    Refer to the `AiiDA complete installation guide <https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#>`_ for full setup details.


Usage
+++++

A quick demo of how to submit a calculation (these require a broker to be setup for daemon to start)::

    verdi daemon start         # make sure the daemon is running
    cd examples/calculations
    verdi run submit_train.py        # submit calculation
    verdi calculation list -a  # check status of calculation

If you have already set up your own aiida_mlip code using
``verdi code setup``, you may want to try the following command::

    mlip-submit  # uses aiida_mlip.cli

Available calculations
++++++++++++++++++++++

These are the available calculations

   * Descriptors
   * GeomOpt
   * MD
   * Singlepoint
   * Train

For more details on the calculations, please refer to the `calculations section <https://stfc.github.io/aiida-mlip/user_guide/calculations.html>`_.
