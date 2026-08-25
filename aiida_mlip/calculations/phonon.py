"""Class to run Phonon calculations."""

from __future__ import annotations

from aiida.common import datastructures
import aiida.common.folders
from aiida.engine import CalcJobProcessSpec
import aiida.engine.processes
from aiida.orm import Bool, Dict, Float, Int, SinglefileData, Str

from aiida_mlip.calculations.base import BaseJanus


class Phonons(BaseJanus):
    """
    Calcjob implementation to run Phonon calculations using the janus-core package.

    Attributes
    ----------
    PHONON_OUTPUT : str
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

    PHONON_OUTPUT = "aiida-phonopy.yml"
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
            default=lambda: Str(cls.PHONON_OUTPUT),
            help="Name of the phonon output file",
        )

        spec.input(
            "supercell",
            valid_type=Str,
            required=False,
            help="The size of supercells used in phonon calculation",
        )
        spec.input(
            "nqpoints",
            valid_type=Int,
            required=False,
            help="Number of q points in band path",
        )
        spec.input(
            "displacement",
            valid_type=Float,
            required=False,
            help="Displacement for numerical derivatives",
        )
        spec.input(
            "no_hdf5",
            valid_type=Bool,
            required=False,
            help="Write force constants to phonopy yaml, rather than separate HDF5.",
        )
        spec.input(
            "dos",
            valid_type=Bool,
            required=False,
            help="Calculate the density of states",
        )
        spec.input(
            "pdos",
            valid_type=Bool,
            required=False,
            help="Calculate the partial density of states",
        )
        spec.input(
            "bands",
            valid_type=Bool,
            required=False,
            help="Calculate the phonon band structure",
        )
        spec.input(
            "symmetrize",
            valid_type=Bool,
            required=False,
            help="Symmetrize force constants",
        )
        spec.input(
            "qpoint_file",
            valid_type=Str,
            required=False,
            help="The file for q-points in phonon calculation",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.ph_parser"

        # Define outputs. The default is a dictionary with the content of the
        # phonon file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )
        spec.output("phonon_output", valid_type=SinglefileData)
        spec.output("force_constants", valid_type=SinglefileData)
        spec.output("dos", valid_type=SinglefileData)
        spec.output("pdos", valid_type=SinglefileData)
        spec.output("band_structure", valid_type=SinglefileData)

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

        # filename when recovering the outputs.
        phonon_filename = (self.inputs.out).value

        # Gather node inputs for use with janus CLI
        supercell = (self.inputs.supercell).value
        displacement = (self.inputs.displacement).value
        nqpoints = (self.inputs.nqpoints).value
        aiida_prefix = "aiida"

        codeinfo.cmdline_params = [
            "phonons",
            *codeinfo.cmdline_params[1:],
            "--file-prefix",
            aiida_prefix,
            "--supercell",
            supercell,
            "--displacement",
            displacement,
        ]

        nohdf5 = (self.inputs.no_hdf5).value
        if nohdf5:
            # force janus-core to write out force constants in yaml
            codeinfo.cmdline_params += [
                "--no-hdf5",
            ]

        dos = (self.inputs.dos).value
        if dos:
            codeinfo.cmdline_params += [
                "--dos",
            ]

        pdos = (self.inputs.pdos).value
        if pdos:
            codeinfo.cmdline_params += [
                "--pdos",
            ]

        symmetrize = (self.inputs.symmetrize).value
        if symmetrize:
            codeinfo.cmdline_params += [
                "--symmetrize",
            ]

        bands = (self.inputs.bands).value
        if bands:
            codeinfo.cmdline_params += [
                "--bands",
            ]
            codeinfo.cmdline_params += ["--n-qpoints", nqpoints]

        calcinfo.retrieve_list.extend(
            [
                phonon_filename,
                "aiida-force_constants.hdf5",
                "aiida-dos.dat",
                "aiida-pdos.dat",
                "aiida-auto_bands.yml.xz",
                # "aiida-bands.hdf5",
            ]
        )

        return calcinfo
