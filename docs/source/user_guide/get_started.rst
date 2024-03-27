===============
Getting started
===============

This page should contain a short guide on what the plugin does and
a short example on how to use the plugin.

Installation
++++++++++++

Use the following commands to install the plugin::

    git clone https://github.com/aiidateam/aiida-mlip .
    cd aiida-mlip
    pip install -e .  # also installs aiida, if missing (but not postgres)
    #pip install -e .[pre-commit,testing] # install extras for more features
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
