"""Class to run Phonon calculations."""

from __future__ import annotations

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Dict, SinglefileData, Str, Bool

from aiida_mlip.calculations.base import BaseJanus


class Phonons(BaseJanus):  # numpydoc ignore=PR01
    """
    Calcjob implementation to run Phonon calculations using mlips and in particular the janus-core package. The routine has employed the singlepoint oarser as a template

    Attributes
    ----------
    phon_output : str
        Default phonon output file name.

    Methods
    -------
    define(spec: CalcJobProcessSpec) -> None:
        Define the process specification, its inputs, outputs and exit codes.
    validate_inputs(value: dict, port_namespace: PortNamespace) -> str | None:
        Check if the inputs are valid.
    prepare_for_submission(folder: Folder) -> CalcInfo:
        Create the input files for the `CalcJob`.
    """

    XYZ_OUTPUT = "aiida-phonopy.yml"
    DEFAULT_SUMMARY_FILE = "phonon-summary.yml"

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

        spec.input(
            "out",
            valid_type=Str,
            required=False,
            default=lambda: Str(cls.XYZ_OUTPUT),
            help="Name of the phonon output file",
        )

        spec.input(
            "properties",
            valid_type=Str,
            required=False,
            help="Properties to calculate",
        )

        
        spec.input(
            "supercell",
            valid_type=Str,
            required=False,
            help="the size of sippercells used in phonon calculation",
        )
        spec.input(
            "minimize",
            valid_type=Bool,   #these need to be aiida.orm types
            required=False,
            help="minimise unit cell prior to phonon calculation",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.ph_parser"

        # Define outputs. The default is a dictionary with the content of the phonon file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )
        spec.output("xyz_output", valid_type=SinglefileData)

        spec.default_output_node = "results_dict"

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

        #print("inputs ", self.inputs)

        # Adding command line params for when we run janus
        # singlepoint is overwriting the placeholder "calculation" from the base.py file

        # The inputs are saved in the node, but we want their value as a string
        xyz_filename = (self.inputs.out).value
        supercell = (self.inputs.supercell).value
        aiida_prefix = "aiida"
        codeinfo.cmdline_params = [
            "phonons",
            *codeinfo.cmdline_params[1:],
            "--no-hdf5", # this is needed to force janus-core to write out force constants in a yaml file
            "--file-prefix",
            aiida_prefix,
            "--supercell",
            supercell,
        ]

        
        # option to minimize the unit cell before phonons
        minimize = (self.inputs.minimize).value
        if minimize:
            codeinfo.cmdline_params += ["--minimize",]

        #properties left in just in case for further expansion
        if "properties" in self.inputs:
            properties = self.inputs.properties.value
            codeinfo.cmdline_params += ["--properties", properties]

        calcinfo.retrieve_list.append(xyz_filename)

        #print("codeinfo ", codeinfo)

        return calcinfo
