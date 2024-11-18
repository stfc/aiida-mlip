===============
Developer guide
===============

Getting started
+++++++++++++++

We recommend `installing poetry <https://python-poetry.org/docs/#installation>`_
for dependency management when developing for ``aiida-mlip``.

This provides a number of useful features, including:

- Dependency management (``poetry [add,update,remove]`` etc.) and organization (groups)
- Storing the versions of all installations in a ``poetry.lock`` file, for reproducible builds
- Improved dependency resolution
- Virtual environment management (optional)
- Building and publishing tools

Dependencies useful for development can then be installed by running::

    poetry install --with pre-commit,dev,docs


Running the tests
+++++++++++++++++

Packages in the ``dev`` dependency group allow tests to be run locally using ``pytest``, by running::                                                                                                                                               pytest -v                                                                                                                                                                                                                                   Alternatively, tests can be run in separate virtual environments using ``tox``::                                                                                                                                                                    tox run -e ALL                                                                                                                                                                                                                              This will run all unit tests for multiple versions of Python, in addition to testing that the pre-commit passes, and that documentation builds, mirroring the automated tests on GitHub.                                                                                                                                                                                Individual components of the ``tox`` test suite can also be run separately, such as running only running the unit tests with Python 3.9::

    tox run -e py39

See the `tox documentation <https://tox.wiki/>`_ for further options.


Automatic coding style checks
+++++++++++++++++++++++++++++

Packages in the ``pre-commit`` dependency group allow automatic code formatting and linting on every commit.

To set this up, run::

    pre-commit install

After this, the `ruff linter <https://docs.astral.sh/ruff/linter/>`_, `ruff formatter <https://docs.astral.sh/ruff/formatter/>`_, and `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ (docstring style validator), will run before every commit.

Rules enforced by ruff are currently set up to be comparable to:

- `black <https://black.readthedocs.io>`_ (code formatter)
- `pylint <https://www.pylint.org/>`_ (linter)
- `pyupgrade <https://github.com/asottile/pyupgrade>`_ (syntax upgrader)
- `isort <https://pycqa.github.io/isort/>`_ (import sorter)
- `flake8-bugbear <https://pypi.org/project/flake8-bugbear/>`_ (bug finder)

The full set of `ruff rules <https://docs.astral.sh/ruff/rules/>`_ are specified by the ``[tool.ruff]`` sections of `pyproject.toml <https://github.com/stfc/aiida-mlip/blob/main/pyproject.toml>`_.

If you ever need to skip these pre-commit hooks, just use::

    git commit -n

You should also keep the pre-commit hooks up to date periodically, with::

    pre-commit autoupdate

Or consider using `pre-commit.ci <https://pre-commit.ci/>`_.


Continuous integration
++++++++++++++++++++++

``aiida-mlip`` comes with a ``.github`` folder that contains continuous integration tests on every commit using `GitHub Actions <https://github.com/features/actions>`_. It will:

#. run all tests
#. build the documentation
#. check coding style and version number (not required to pass by default)


Building the documentation
++++++++++++++++++++++++++

 #. Install the ``docs`` extra::

        pip install -e .[docs]

 #. Edit the individual documentation pages::

        docs/source/index.rst
        docs/source/developer_guide/index.rst
        docs/source/user_guide/index.rst
        docs/source/user_guide/get_started.rst
        docs/source/user_guide/tutorial.rst

 #. Use `Sphinx`_ to generate the html documentation::

        cd docs
        make

Check the result by opening ``build/html/index.html`` in your browser.

Publishing the documentation
++++++++++++++++++++++++++++

Once you're happy with your documentation, it's easy to host it online on ReadTheDocs_:

 #. Create an account on ReadTheDocs_

 #. Import your ``aiida-mlip`` repository (preferably using ``aiida-mlip`` as the project name)

The documentation is now available at `aiida-mlip.readthedocs.io <http://aiida-mlip.readthedocs.io/>`_.

PyPI release
++++++++++++

Your plugin is ready to be uploaded to the `Python Package Index <https://pypi.org/>`_.
Just register for an account and use `flit <https://flit.readthedocs.io/en/latest/upload.html>`_::

    pip install flit
    flit publish

After this, you (and everyone else) should be able to::

    pip install aiida-mlip

You can also enable *automatic* deployment of git tags to the python package index:
simply generate a `PyPI API token <https://pypi.org/help/#apitoken>`_ for your PyPI account and add it as a secret to your GitHub repository under the name ``pypi_token`` (Go to Settings -> Secrets).


.. _ReadTheDocs: https://readthedocs.org/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
