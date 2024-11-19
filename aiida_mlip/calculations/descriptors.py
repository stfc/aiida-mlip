"""Class to run descriptors calculations."""

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Bool

from aiida_mlip.calculations.singlepoint import Singlepoint


class Descriptors(Singlepoint):  # numpydoc ignore=PR01
    """
    Calcjob implementation to calculate MLIP descriptors.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

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

        # Define inputs

        # Remove unused singlepoint input
        del spec.inputs["properties"]

        spec.input(
            "invariants_only",
            valid_type=Bool,
            required=False,
            help="Only calculate invariant descriptors.",
        )

        spec.input(
            "calc_per_element",
            valid_type=Bool,
            required=False,
            help="Calculate mean descriptors for each element.",
        )

        spec.input(
            "calc_per_atom",
            valid_type=Bool,
            required=False,
            help="Calculate descriptors for each atom.",
        )

        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "mlip.descriptors_parser"

    # pylint: disable=too-many-locals
    def prepare_for_submission(
        self, folder: aiida.common.folders.Folder
    ) -> datastructures.CalcInfo:
        """
        Create the input files for the `Calcjob`.

        Parameters
        ----------
        folder : aiida.common.folders.Folder
            Folder where the calculation is run.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """
        # Call the parent class method to prepare common inputs
        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        # Adding command line params for when we run janus
        # descriptors is overwriting the placeholder "calculation" from the base.py file
        codeinfo.cmdline_params[0] = "descriptors"

        cmdline_options = {}
        if "invariants_only" in self.inputs:
            cmdline_options["invariants-only"] = self.inputs.invariants_only.value
        if "calc_per_element" in self.inputs:
            cmdline_options["calc-per-element"] = self.inputs.calc_per_element.value
        if "calc_per_atom" in self.inputs:
            cmdline_options["calc-per-atom"] = self.inputs.calc_per_atom.value

        for flag, value in cmdline_options.items():
            if isinstance(value, bool):
                # Add boolean flags without value if True
                if value:
                    codeinfo.cmdline_params.append(f"--{flag}")
            else:
                codeinfo.cmdline_params += [f"--{flag}", value]

        return calcinfo
