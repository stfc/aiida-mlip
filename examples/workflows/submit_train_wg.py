"""Example submission for hts workgraph."""

from pathlib import Path

from aiida.orm import Dict, load_code

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.workflows.training_workgraph import TrainWorkGraph

folder_path = Path("/work4/scd/scarf1228/prova_train_workgraph/")
code = load_code("qe-7.1@scarf")
inputs = {
    "base": {
        "settings": Dict({"GAMMA_ONLY": True}),
        "pw": {
            "parameters": Dict(
                {
                    "CONTROL": {
                        "calculation": "vc-relax",
                        "nstep": 1200,
                        "etot_conv_thr": 1e-05,
                        "forc_conv_thr": 1e-04,
                    },
                    "SYSTEM": {
                        "ecutwfc": 500,
                        "input_dft": "PBE",
                        "nspin": 1,
                        "occupations": "smearing",
                        "degauss": 0.001,
                        "smearing": "m-p",
                    },
                    "ELECTRONS": {
                        "electron_maxstep": 1000,
                        "scf_must_converge": False,
                        "conv_thr": 1e-08,
                        "mixing_beta": 0.25,
                        "diago_david_ndim": 4,
                        "startingpot": "atomic",
                        "startingwfc": "atomic+random",
                    },
                    "IONS": {
                        "ion_dynamics": "bfgs",
                    },
                    "CELL": {
                        "cell_dynamics": "bfgs",
                        "cell_dofree": "ibrav",
                    },
                }
            ),
            "code": code,
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 4,
                        "num_mpiprocs_per_machine": 32,
                    },
                    "max_wallclock_seconds": 48 * 60 * 60,
                },
            },
        },
    },
    "base_final_scf": {
        "pw": {
            "parameters": Dict(
                {
                    "CONTROL": {
                        "calculation": "scf",
                        "tprnfor": True,
                    },
                    "SYSTEM": {
                        "ecutwfc": 70,
                        "ecutrho": 650,
                        "input_dft": "PBE",
                        "occupations": "smearing",
                        "degauss": 0.001,
                        "smearing": "m-p",
                    },
                    "ELECTRONS": {
                        "conv_thr": 1e-10,
                        "mixing_beta": 0.25,
                        "diago_david_ndim": 4,
                        "startingpot": "atomic",
                        "startingwfc": "atomic+random",
                    },
                }
            ),
            "code": code,
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 32,
                    },
                    "max_wallclock_seconds": 3 * 60 * 60,
                },
            },
        },
    },
}
janusconfigfile_path = "/work4/scd/scarf1228/prova_train_workgraph/mlip_train.yml"
janusconfigfile = JanusConfigfile(file=janusconfigfile_path)

TrainWorkGraph(folder_path, inputs, janusconfigfile)
