[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

# aiida-mlip
![logo][logo]

machine learning interatomic potentials aiida plugin

## Features (in development)

- [x] Supports multiple MLIPs
  - MACE
  - M3GNET
  - CHGNET
- [x] Single point calculations
- [x] Geometry optimisation
- [x] Molecular Dynamics:
  - NVE
  - NVT (Langevin(Eijnden/Ciccotti flavour) and Nosé-Hoover (Melchionna flavour))
  - NPT (Nosé-Hoover (Melchiona flavour))
- [x] Training MLIPs
  - MACE
- [x] Fine tuning MLIPs
  - MACE
- [x] MLIP descriptors
  - MACE

The code relies heavily on [janus-core](https://github.com/stfc/janus-core), which handles mlip calculations using ASE.


# Getting Started

## Installation
Create a Python [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) and activate it to install aiida-mlip

```shell
pip install aiida-mlip
verdi plugin list aiida.calculations
```
The last command should show a list of AiiDA pre-installed calculations and the aiida-mlip plugin calculations (mlip.opt, mlip.sp)
```
Registered entry points for aiida.calculations:
* core.arithmetic.add
* core.templatereplacer
* core.transfer
* mlip.opt
* mlip.sp
* mlip.md
* mlip.train
* mlip.descriptors
```

Aiida-mlip should be ready to run some notebooks at this stage. However, to have full functionality we reccomend configuring aiida-mlip by creating a profile and setting up a broker.

## AiiDA Configuration

Now that aiida-mlip plugin has been installed, we can setup the environment to run calculations:

1. Install RabbitMQ ([ Link ](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#rabbitmq))
2. Run:
```shell
verdi presto #Sets up profile
```
`verdi presto` is a quick and simple way to setup the daemon to run some calculations, setting up a profile 'presto'; which configures the computer, broker (i.e. RabbitMQ) and database. [Aiida docs](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#) go over a more detailed proccess to setup a profile.

## Usage

The example folder provides scripts to submit calculations in the calculations folder, and tutorials in jupyter notebook format in the tutorials folder.

A quick demo of how to submit a calculation using the provided example files:

You need to create a [code](https://aiida.readthedocs.io/projects/aiida-core/en/stable/howto/run_codes.html#how-to-create-a-code) for Janus to be recognised as a code by aiida

```shell
verdi daemon start     # make sure the daemon is running
cd examples/calculations
verdi run submit_singlepoint.py "janus@localhost" --struct "path/to/structure" --architecture mace --model "/path/to/model"    # run singlepoint calculation
verdi run submit_geomopt.py "janus@localhost" --struct "path/to/structure" --model "path/to/model" --steps 5 --opt_cell_fully True # run geometry optimisation
verdi run submit_md.py "janus@localhost" --struct "path/to/structure" --model "path/to/model" --ensemble "nve" --md_dict_str "{'temp':300,'steps':4,'traj-every':3,'stats-every':1}" # run molecular dynamics

verdi process list -a  # check record of calculation
```
Models can be trained by using the Train calcjob. In that case the needed inputs are a config file containig the path to train, test and validation xyz file and other optional parameters. Running
```shell
verdi run submit_train.py
```
a model will be trained using the provided example config file and xyz files (can be found in the tests folder)


## Development

We recommend installing uv for dependency management when developing for `aiida-mlip`:


1. Install [uv](https://docs.astral.sh/uv/getting-started/installation)
2. Setup [PostgreSQL](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#core-psql-dos)
3. Install `aiida-mlip` with dependencies in a virtual environment:

```shell
git clone https://github.com/stfc/aiida-mlip
cd aiida-mlip
uv sync --extra mace # Create a virtual environment and install dependencies with mace for tests
source .venv/bin/activate
pre-commit install  # Install pre-commit hooks
pytest -v  # Discover and run all tests
```
See the [developer guide](https://stfc.github.io/aiida-mlip/developer_guide/index.html) for more information.


## Repository contents

* [`.github/`](.github/): [Github Actions](https://github.com/features/actions) configuration
  * [`ci.yml`](.github/workflows/ci.yml): runs tests, checks test coverage and builds documentation at every new commit
  * [`publish-on-pypi.yml`](.github/workflows/publish-on-pypi.yml): automatically deploy git tags to PyPI - just generate a [PyPI API token](https://pypi.org/help/#apitoken) for your PyPI account and add it to the `pypi_token` secret of your github repository
  * [`docs.yml`](.github/workflows/docs.yml): builds and deploys the documentation
* [`aiida_mlip/`](aiida_mlip/): The main source code of the plugin package
  * [`data/`](aiida_mlip/data/): Plugin `Data` classes
    * [`model.py`](aiida_mlip/data/model.py) `ModelData` class to save mlip models as AiiDA data types
  * [`calculations/`](aiida_mlip/calculations/): Plugin `Calcjob` classes
    * [`base.py`](aiida_mlip/calculations/base.py): Base `Calcjob` class for other calculations
    * [`singlepoint.py`](aiida_mlip/calculations/singlepoint.py): `Calcjob` class to run single point calculations using mlips
    * [`geomopt.py`](aiida_mlip/calculations/geomopt.py): `Calcjob` class to perform geometry optimization using mlips
    * [`md.py`](aiida_mlip/calculations/md.py): `Calcjob` class to perform molecular dynamics using mlips
    * [`descriptors.py`](aiida_mlip/calculations/descriptors.py): `Calcjob` class to calculate MLIP descriptors
  * [`parsers/`](aiida_mlip/parsers/): `Parsers` for the calculations
    * [`base_parser.py`](aiida_mlip/parsers/base_parser.py): Base `Parser` for all calculations.
    * [`sp_parser.py`](aiida_mlip/parsers/sp_parser.py): `Parser` for `Singlepoint` calculation.
    * [`opt_parser.py`](aiida_mlip/parsers/opt_parser.py): `Parser` for `Geomopt` calculation.
    * [`md_parser.py`](aiida_mlip/parsers/md_parser.py): `Parser` for `MD` calculation.
    * [`train_parser.py`](aiida_mlip/parsers/train_parser.py): `Parser` for `Train` calculation.
    * [`descriptors_parser.py`](aiida_mlip/parsers/descriptors_parser.py): `Parser` for `Descriptors` calculation.
  * [`helpers/`](aiida_mlip/helpers/): `Helpers` to run calculations.
  * [`workflows/`](aiida_mlip/workflows/): `WorkGraphs` or `WorkChains` for common workflows with mlips.
    * [`ht_workgraph.py`](aiida_mlip/workflows/ht_workgraph.py): A `WorkGraph` to run high-throughput optimisations.
* [`docs/`](docs/source/): Code documentation
  * [`apidoc/`](docs/source/apidoc/): API documentation
  * [`developer_guide/`](docs/source/developer_guide/): Documentation for developers
  * [`user_guide/`](docs/source/user_guide/): Documentation for users
  * [`images/`](docs/source/images/): Logos etc used in the documentation
* [`examples/`](examples/): Examples for submitting calculations using this plugin
  * [`tutorials/`](examples/tutorials/): Jupyter notebooks with tutorials for running calculations and other files that are used in the tutorial
  * [`calculations/`](examples/calculations/): Scripts for submitting calculations
    * [`submit_singlepoint.py`](examples/calculations/submit_singlepoint.py): Script for submitting a singlepoint calculation
    * [`submit_geomopt.py`](examples/calculations/submit_geomopt.py): Script for submitting a geometry optimisation calculation
    * [`submit_md.py`](examples/calculations/submit_md.py): Script for submitting a molecular dynamics calculation
    * [`submit_train.py`](examples/calculations/submit_train.py): Script for submitting a train calculation.
    * [`submit_descriptors.py`](examples/calculations/submit_descriptors.py): Script for submitting a descriptors calculation.
  * [`workflows/`](examples/workflows/): Scripts for submitting workflows
    * [`submit_ht_workgraph.py`](examples/workflows/submit_ht_workgraph.py): Script for submitting a high-throughput WorkGraph for singlepoint calculation.
* [`tests/`](tests/): Basic regression tests using the [pytest](https://docs.pytest.org/en/latest/) framework (submitting a calculation, ...). Install `pip install -e .[testing]` and run `pytest`.
  * [`conftest.py`](tests/conftest.py): Configuration of fixtures for [pytest](https://docs.pytest.org/en/latest/)
  * [`calculations/`](tests/calculations): Calculations
    * [`test_singlepoint.py`](tests/calculations/test_singlepoint.py): Test `SinglePoint` calculation
    * [`test_geomopt.py`](tests/calculations/test_geomopt.py): Test `Geomopt` calculation
    * [`test_md.py`](tests/calculations/test_md.py): Test `MD` calculation
    * [`test_train.py`](tests/calculations/test_train.py): Test `Train` calculation
    * [`test_descriptors.py`](tests/calculations/test_descriptors.py): Test `Descriptors` calculation
  * [`data/`](tests/data): Data
    * [`test_model.py`](tests/data/test_model.py): Test `ModelData` type
    * [`test_config.py`](tests/data/test_config.py): Test `JanusConfigfile` type
  * [`workflows/`](tests/workflows): Workflows
    * [`test_ht.py`](tests/workflows/test_ht.py): Test high throughput workgraph.
* [`.gitignore`](.gitignore): Telling git which files to ignore
* [`.pre-commit-config.yaml`](.pre-commit-config.yaml): Configuration of [pre-commit hooks](https://pre-commit.com/) that sanitize coding style and check for syntax errors. Enable via `pip install -e .[pre-commit] && pre-commit install`
* [`LICENSE`](LICENSE): License for the plugin
* [`README.md`](README.md): This file
* [`tox.ini`](tox.ini): File to set up tox
* [`pyproject.toml`](pyproject.toml): Python package metadata for registration on [PyPI](https://pypi.org/) and the [AiiDA plugin registry](https://aiidateam.github.io/aiida-registry/) (including entry points)


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
[logo]: https://raw.githubusercontent.com/stfc/aiida-mlip/main/docs/source/images/aiida-mlip-100.png
