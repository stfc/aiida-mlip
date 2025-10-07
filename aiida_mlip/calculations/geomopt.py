"""Class to run geom opt calculations."""

from __future__ import annotations

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    SinglefileData,
    StructureData,
    TrajectoryData,
)
from aiida.orm.utils.managers import NodeLinksManager
from plumpy.utils import AttributesFrozendict

from aiida_mlip.calculations.singlepoint import Singlepoint
from aiida_mlip.helpers.converters import kwarg_to_param


class GeomOpt(Singlepoint):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run geometry optimisation calculations using mlips.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    DEFAULT_TRAJ_FILE = "aiida-traj.xyz"
    DEFAULT_SUMMARY_FILE = "geomopt-summary.yml"

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

        # Additional inputs for geometry optimisation
        spec.input(
            "opt_cell_fully",
            valid_type=Bool,
            required=False,
            help="Fully optimise the cell vectors, angles, and atomic positions",
        )
        spec.input(
            "opt_cell_lengths",
            valid_type=Bool,
            required=False,
            help="Optimise cell vectors, as well as atomic positions",
        )
        spec.input(
            "fmax",
            valid_type=Float,
            required=False,
            help="Maximum force for convergence",
        )

        spec.input(
            "steps",
            valid_type=Int,
            required=False,
            help="Number of optimisation steps",
        )

        spec.input(
            "minimize_kwargs",
            valid_type=Dict,
            required=False,
            help="All other keyword arguments to pass to geometry optimizer",
        )

        spec.input(
            "pressure",
            valid_type=Float,
            required=False,
            help="Scalar pressure when optimizing cell geometry, in GPa.",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.opt_parser"

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
            Folder where the calculation is run.

        Returns
        -------
        aiida.common.datastructures.CalcInfo
            An instance of `aiida.common.datastructures.CalcInfo`.
        """
        # Call the parent class method to prepare common inputs
        calcinfo = super().prepare_for_submission(folder)
        codeinfo = calcinfo.codes_info[0]

        minimize_kwargs = self.set_minimize_kwargs(self.inputs)

        geom_opt_cmdline = {
            "minimize-kwargs": minimize_kwargs,
            "write-traj": True,
        }
        if "opt_cell_fully" in self.inputs:
            geom_opt_cmdline["opt-cell-fully"] = self.inputs.opt_cell_fully.value
        if "opt_cell_lengths" in self.inputs:
            geom_opt_cmdline["opt-cell-lengths"] = self.inputs.opt_cell_lengths.value
        if "fmax" in self.inputs:
            geom_opt_cmdline["fmax"] = self.inputs.fmax.value
        if "steps" in self.inputs:
            geom_opt_cmdline["steps"] = self.inputs.steps.value
        if "pressure" in self.inputs:
            geom_opt_cmdline["pressure"] = self.inputs.pressure.value

        # Adding command line params for when we run janus
        # 'geomopt' is overwriting the placeholder "calculation" from the base.py file
        codeinfo.cmdline_params = [
            "geomopt",
            *codeinfo.cmdline_params[1:],
            *kwarg_to_param(geom_opt_cmdline),
        ]

        calcinfo.retrieve_list.append(minimize_kwargs["traj_kwargs"]["filename"])

        return calcinfo

    @classmethod
    def set_minimize_kwargs(
        cls, inputs: AttributesFrozendict | NodeLinksManager
    ) -> dict[str, dict[str, str]]:
        """
        Set minimize kwargs from CalcJob inputs.

        Parameters
        ----------
        inputs : x
            CalcJob inputs.

        Returns
        -------
        dict[str, dict[str, str]]
            Set minimize_kwargs dict with trajectory filename extracted from `traj`,
            the config file, or set as the default.
        """
        if "minimize_kwargs" in inputs:
            minimize_kwargs = inputs.minimize_kwargs.get_dict()
        elif "config" in inputs:
            minimize_kwargs = inputs.config.as_dictionary.get("minimize_kwargs", {})
        else:
            minimize_kwargs = {}

        minimize_kwargs.setdefault("traj_kwargs", {})
        minimize_kwargs["traj_kwargs"].setdefault("filename", cls.DEFAULT_TRAJ_FILE)

        return minimize_kwargs
