===============
Getting started
===============

This page should contain a short guide on what the plugin does and
a short example on how to use the plugin.

Installation
++++++++++++

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

    verdi quicksetup  # better to set up a new profile
    verdi plugin list aiida.calculations  # should now show your calculation plugins

Then use ``verdi code setup`` with the ``janus`` input plugin
to set up an AiiDA code for aiida-mlip.


Usage
+++++

A quick demo of how to submit a calculation::

    verdi daemon start         # make sure the daemon is running
    cd examples
    verdi run test_submit.py        # submit test calculation
    verdi calculation list -a  # check status of calculation

If you have already set up your own aiida_mlip code using
``verdi code setup``, you may want to try the following command::

    mlip-submit  # uses aiida_mlip.cli

Available calculations
++++++++++++++++++++++

.. aiida-calcjob:: Singlepoint
    :module: aiida_mlip.calculations.singlepoint
