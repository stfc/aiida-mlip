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
    If you have not set up the RabbitMQ message broker, you should be now able to ``run`` processes,
    as demonstrated in the `tutorial notebooks <https://github.com/stfc/aiida-mlip/tree/main/examples/tutorials>`_,
    but cannot ``submit`` processes.

    The PostgreSQL database is not configured by default.
    Please refer to the `Aiida documentation <https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html#quick-install-limitations>`_
    for more details on the limitations of not setting up a broker or PostgreSQL.

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

These are the available calculations::

    class aiida_mlip.calculations.descriptors.Descriptors(*args: Any, **kwargs: Any)
    class aiida_mlip.calculations.geomopt.GeomOpt(*args: Any, **kwargs: Any)
    class aiida_mlip.calculations.md.MD(*args: Any, **kwargs: Any)
    class aiida_mlip.calculations.singlepoint.Singlepoint(*args: Any, **kwargs: Any)
    class aiida_mlip.calculations.train.Train(*args: Any, **kwargs: Any)

For more details on the calculations, please refer to the `calculations section <https://stfc.github.io/aiida-mlip/user_guide/calculations.html>`_.
