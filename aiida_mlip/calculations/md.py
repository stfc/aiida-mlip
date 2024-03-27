"""Class to run md calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import (
    Dict,
    SinglefileData,
    Str,
    StructureData,
    TrajectoryData,
)

from aiida_mlip.calculations.singlepoint import Singlepoint


class MD(Singlepoint):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run geometry MD calculations using mlips.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    _DEFAULT_TRAJ_FILE = "aiida-traj.xyz"

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

        # Additional inputs for molecula dynamics
        spec.input(
            "ensemble",
            valid_type=Str,
            required=True,
            help="Name for thermodynamic ensemble",
        )

        spec.input(
            "md_dict",
            valid_type=Dict,
            required=False,
            default=lambda: Dict({}),
            help="Keywords for molecular dynamics",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "janus.md_parser"

        spec.output("stat_file", valid_type=SinglefileData)
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

        md_dict = self.inputs.md_dict.get_dict()

        ensemble = self.inputs.ensemble.value.lower()
        codeinfo.cmdline_params[0] = "md"
        codeinfo.cmdline_params.append(f"--ensemble : {ensemble}")

        for flag, value in md_dict.items():
            if isinstance(value, bool):
                # Add boolean flags without value if True
                if value:
                    codeinfo.cmdline_params.append(f"--{flag}")
            else:
                codeinfo.cmdline_params += [f"--{flag}", value]

        calcinfo.retrieve_list.append(self.inputs.traj.value)

        return calcinfo
