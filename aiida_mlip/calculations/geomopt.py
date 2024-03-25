"""Class to run geom opt calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    SinglefileData,
    Str,
    StructureData,
    TrajectoryData,
)

from aiida_mlip.calculations.singlepoint import Singlepoint


class GeomOpt(Singlepoint):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run geometry optimization calculations using mlips.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    _DEFAULT_TRAJ_FILE = "aiida-traj.xyz"
    _DEFAULT_RESTART_FILE = "restart_file.txt"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the process specification, its inputs, outputs and exit codes.

        Parameters
        ----------
        spec : `aiida.engine.CalcJobProcessSpec`
            The calculation job process spec to define.
        """
        super().define(spec)

        # Additional inputs for geometry optimization
        spec.input(
            "traj",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls._DEFAULT_TRAJ_FILE),
            help="Path to save optimization frames to",
        )
        spec.input(
            "fully_opt",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help="Fully optimize the cell vectors, angles, and atomic positions",
        )
        spec.input(
            "vectors_only",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help="Optimize cell vectors, as well as atomic positions",
        )
        spec.input(
            "max_force",
            valid_type=Float,
            required=False,
            default=lambda: Float(0.1),
            help="Maximum force for convergence",
        )

        spec.input(
            "steps",
            valid_type=Int,
            required=False,
            default=lambda: Int(1000),
            help="Number of optimisation steps",
        )

        spec.input(
            "opt_kwargs",
            valid_type=Dict,
            required=False,
            default=lambda: Dict({}),
            help="Other optimisation keywords",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "janus.opt_parser"

        spec.output("traj_file", valid_type=SinglefileData)
        spec.output("traj_output", valid_type=TrajectoryData)
        spec.output("final_structure", valid_type=StructureData)

    def prepare_for_submission(
        self, folder: aiida.common.folders.Folder
    ) -> datastructures.CalcInfo:
        """
        Create the input files for the `Calcjob`.

        Parameters
        ----------
        folder : aiida.common.folders.Folder
            An `aiida.common.folders.Folder` to temporarily write files on disk.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """

        # Call the parent class method to prepare common inputs
        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        opt_kwargs = self.inputs.opt_kwargs.get_dict()

        geom_opt_cmdline = {
            "traj": self.inputs.traj.value,
            "fully-opt": self.inputs.fully_opt.value,
            "vectors-only": self.inputs.vectors_only.value,
            "max-force": self.inputs.max_force.value,
            "steps": self.inputs.steps.value,
            "opt-kwargs": opt_kwargs,
        }

        codeinfo.cmdline_params[0] = "geomopt"

        for flag, value in geom_opt_cmdline.items():
            if isinstance(value, bool):
                # Add boolean flags without value if True
                if value:
                    codeinfo.cmdline_params.append(f"--{flag}")
            else:
                codeinfo.cmdline_params += [f"--{flag}", value]

        calcinfo.retrieve_list.append(self.inputs.traj.value)

        return calcinfo
