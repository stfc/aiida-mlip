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
verdi presto #Sets up profile
verdi plugin list aiida.calculations
```
The last command should show a list of AiiDA pre-installed calculations and the aiida-mlip plugin calculations:
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

Aiida-mlip should be ready to run some notebooks at this stage . However, to have [full functionality](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_quick.html#quick-install-limitations) we recommend configuring aiida-mlip by creating a profile and setting up a broker.

## AiiDA Configuration

1. Install [RabbitMQ](https://aiida.readthedocs.io/projects/aiida-core/en/stable/installation/guide_complete.html#rabbitmq)
2. Run:
```shell
verdi presto #Sets up profile and broker for daemon to run
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

We recommend installing uv for dependency management when developing for `aiida-mlip` and setting up PostgreSQL:


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
