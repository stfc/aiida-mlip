[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

# aiida-mlip

machine learning interatomic potentials aiida plugin

This plugin is the default output of the
[AiiDA plugin cutter](https://github.com/aiidateam/aiida-plugin-cutter),
intended to help developers get started with their AiiDA plugins.

## Repository contents

* [`.github/`](.github/): [Github Actions](https://github.com/features/actions) configuration
  * [`ci.yml`](.github/workflows/ci.yml): runs tests, checks test coverage and builds documentation at every new commit
  * [`publish-on-pypi.yml`](.github/workflows/publish-on-pypi.yml): automatically deploy git tags to PyPI - just generate a [PyPI API token](https://pypi.org/help/#apitoken) for your PyPI account and add it to the `pypi_token` secret of your github repository
  * [`docs.yml`](.github/workflows/docs.yml): builds and deploys the documentasion
* [`aiida_mlip/`](aiida_mlip/): The main source code of the plugin package
  * [`data/`](aiida_mlip/data/): Plugin `Data` classes
    * [`model.py/`](aiida_mlip/data/model.py) `ModelData` class to save mlip models as AiiDA data types
  * [`calculations/`](aiida_mlip/calculations/): Plugin `Calcjob` classes
    * [`singlepoint.py](aiida_mlip/calculations/singlepoint.py ): `Calcjob` class to run single point calculations using mlips
  * [`parsers.py`](aiida_mlip/parsers.py): `Parser` for the `Singlepoint` calculation
* [`docs/`](docs/source/): Code documentation
  * [`apidoc/`](docs/source/apidoc/): API documentation
  * [`developer_guide/`](docs/source/developer_guide/): Documentation for developers
  * [`user_guide`](docs/source/user_guide/): Documentation for users
  * [`images`](docs/source/images/): Logos etc used in the documentation
* [`examples/`](examples/): Examples for submitting calculations using this plugin
  *[`calculations/submit_singlepoint.py`](examples/calculations/submit_singlepoint.py): Script for submitting a singlepoint calculation
* [`tests/`](tests/): Basic regression tests using the [pytest](https://docs.pytest.org/en/latest/) framework (submitting a calculation, ...). Install `pip install -e .[testing]` and run `pytest`.
  * [`conftest.py`](conftest.py): Configuration of fixtures for [pytest](https://docs.pytest.org/en/latest/)
  * [`calculations/`](tests/calculations): Test `SinglePoint` calculation
    * [`test_singlepoint.py`](tests/calculations/test_singlepoint.py): Calculations
  * [`data/`](tests/data): `ModelData`
    * [`test_model.py`](tests/data/test_model.py): Test `ModelData` type
* [`.gitignore`](.gitignore): Telling git which files to ignore
* [`.pre-commit-config.yaml`](.pre-commit-config.yaml): Configuration of [pre-commit hooks](https://pre-commit.com/) that sanitize coding style and check for syntax errors. Enable via `pip install -e .[pre-commit] && pre-commit install`
* [`LICENSE`](LICENSE): License for the plugin
* [`README.md`](README.md): This file
* [`tox.ini`](tox.ini): File to set up tox
* [`pyproject.toml`](pyproject.toml): Python package metadata for registration on [PyPI](https://pypi.org/) and the [AiiDA plugin registry](https://aiidateam.github.io/aiida-registry/) (including entry points)



## Features

 * Save MLIP models using `ModelData`

## Installation

```shell
pip install aiida-mlip
verdi quicksetup  # better to set up a new profile
verdi plugin list aiida.calculations  # should now show your calclulation plugins
```


## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:
```shell
verdi daemon start     # make sure the daemon is running
cd examples
verdi run submit_singlepoint.py "janus@localhost" --calctype "singlepoint"  --architecture mace_mp --model "~./cache/mlips/mace_mp/46jrkm3v"       # run test calculation
verdi process list -a  # check record of calculation
```

## Development

1. Install [poetry](https://python-poetry.org/docs/#installation)
2. (Optional) Create a virtual environment
3. Install `aiida-mlip` with dependencies:

```shell
git clone https://github.com/stfc/aiida-mlip
cd aiida-mlip
pip install --upgrade pip
poetry install --with pre-commit,dev,docs  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

See the [developer guide](https://stfc.github.io/aiida-mlip/developer_guide/index.html) for more information.

## License

[BSD 3-Clause License](LICENSE)

## Funding

Contributors to this project were funded by

[![PSDI](https://raw.githubusercontent.com/stfc/aiida-mlip/main/docs/source/images/psdi-100.webp)](https://www.psdi.ac.uk/)
[![ALC](https://raw.githubusercontent.com/stfc/aiida-mlip/main/docs/source/images/alc-100.webp)](https://adalovelacecentre.ac.uk/)
[![CoSeC](https://raw.githubusercontent.com/stfc/aiida-mlip/main/docs/source/images/cosec-100.webp)](https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx)


[ci-badge]: https://github.com/stfc/aiida-mlip/workflows/ci/badge.svg
[ci-link]: https://github.com/stfc/aiida-mlip/actions
[cov-badge]: https://coveralls.io/repos/github/stfc/aiida-mlip/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/stfc/aiida-mlip?branch=main
[docs-badge]: https://github.com/stfc/aiida-mlip/actions/workflows/docs.yml/badge.svg
[docs-link]: https://stfc.github.io/aiida-mlip/
[pypi-badge]: https://badge.fury.io/py/aiida-mlip.svg
[pypi-link]: https://badge.fury.io/py/aiida-mlip
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
[doi-link]: https://zenodo.org/badge/latestdoi/750834002
[doi-badge]: https://zenodo.org/badge/750834002.svg
