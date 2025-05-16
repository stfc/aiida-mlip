===============
Developer guide
===============

Getting started
+++++++++++++++

We recommend `installing uv <https://docs.astral.sh/uv/getting-started/installation/>`_
for dependency management when developing for ``aiida-mlip``.

This provides a number of useful features, including:

- `Dependency management <https://docs.astral.sh/uv/concepts/projects/dependencies/>`_ (``uv [add,remove]`` etc.) and organization (`groups <https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups>`_)

- Storing the versions of all installations in a `uv.lock <https://docs.astral.sh/uv/concepts/projects/sync/>`_ file, for reproducible builds

- Improved `dependency resolution <https://docs.astral.sh/uv/concepts/resolution/>`_

- Virtual environment management

- `Building and publishing <https://docs.astral.sh/uv/guides/publish/>`_ tools

  * Currently, an external build backend, such as `pdm <https://pypi.org/project/pdm-backend>`_, is required


After cloning the repository, dependencies useful for development can then be installed by running::

    uv sync -p 3.12 --extra mace -U
    source .venv/bin/activate


Using uv
++++++++

``uv`` manages a `persistent environment <https://docs.astral.sh/uv/concepts/projects/layout/#the-project-environment>`_
with the project and its dependencies in a ``.venv`` directory, adjacent to ``pyproject.toml``. This will be created automatically as needed.

``uv`` provides two separate APIs for managing your Python project and environment.

``uv pip`` is designed to resemble the ``pip`` CLI, with similar commands (``uv pip install``,  ``uv pip list``, ``uv pip tree``, etc.),
and is slightly lower level. `Compared with pip <https://docs.astral.sh/uv/pip/compatibility/>`_,
``uv`` tends to be stricter, but in most cases ``uv pip`` could be used in place of ``pip``.

``uv add``, ``uv run``, ``uv sync``, and ``uv lock`` are known as "project APIs", and are slightly higher level.
These commands interact with (and require) ``pyproject.toml``, and ``uv`` will ensure your environment is in-sync when they are called,
including creating or updating a `lockfile <https://docs.astral.sh/uv/concepts/projects/sync/>`_,
a universal resolution that is `portable across platforms <https://docs.astral.sh/uv/concepts/resolution/#universal-resolution>`_.

When developing for ``aiida-mlip``, it is usually recommended to use project commands, as described in `Getting started`_
rather than using ``uv pip install`` to modify the project environment manually.

.. tip::

    ``uv`` will detect and use Python versions available on your system,
    but can also be used to `install Python automtically <https://docs.astral.sh/uv/guides/install-python/>`_.
    The desired Python version can be specified when running project commands with the ``--python``/``-p`` option.


For further information, please refer to the `documentation <https://docs.astral.sh/uv/>`_.

Setting up PostgreSQL
+++++++++++++++++++++

``aiida-mlip`` requires a PostgreSQL database to be set up for the tests to run successfully.

PostgreSQL can be installed outside the virtual environment::

    sudo apt install postgresql

The `Ubuntu Server <https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/index.html>`_ docs go over installing PostgreSQL on Ubuntu.
For other operating systems, please refer to the `PostgreSQL documentation <https://www.postgresql.org/download/>`_.

Then for specific instructions on setting up PostgreSQL for AiiDA, please refer to the `AiiDA documentation <https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#core-psql-dos>`_.


Running the tests
+++++++++++++++++

Packages in the ``dev`` dependency group allow tests to be run locally using ``pytest``, by running::

    pytest -v

.. note::

    MACE must be installed for tests to run successfully. PostgreSQL must also be installed and running.


Alternatively, tests can be run in separate virtual environments using ``tox``::

    tox run -e ALL

This will run all unit tests for multiple versions of Python, in addition to testing that the pre-commit passes, and that documentation builds, mirroring the automated tests on GitHub.

Individual components of the ``tox`` test suite can also be run separately, such as running only running the unit tests with Python 3.12::

    tox run -e py312

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


Building the documentation
++++++++++++++++++++++++++

Packages in the ``docs`` dependency group install `Sphinx <https://www.sphinx-doc.org>`_
and other Python packages required to build ``aiida-mlip``'s documentation.

Individual individual documentation pages can be edited directly::

        docs/source/index.rst
        docs/source/developer_guide/index.rst
        docs/source/user_guide/index.rst
        docs/source/user_guide/get_started.rst
        docs/source/user_guide/tutorial.rst


``Sphinx`` can then be used to generate the html documentation::

        cd docs
        make clean; make html


Check the result by opening ``build/html/index.html`` in your browser.


Continuous integration
++++++++++++++++++++++

``aiida-mlip`` comes with a ``.github`` folder that contains continuous integration tests
on every commit using `GitHub Actions <https://github.com/features/actions>`_. It will:

#. Run all tests
#. Build the documentation
#. Check coding style
