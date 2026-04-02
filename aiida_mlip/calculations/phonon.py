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
            help="the size of supercells used in phonon calculation",
        )
        spec.input(
            "nqpoints",
            valid_type=Int,
            required=False,
            help="number of q points in band path",
        )
        spec.input(
            "fmax",
            valid_type=Float,
            required=False,
            help="fmax for geometry optimisation",
        )
        spec.input(
            "displacement",
            valid_type=Float,
            required=False,
            help="displacement for numerical derivatives",
        )
        spec.input(
            "minimize",
            valid_type=Bool,
            required=False,
            help="minimise unit cell prior to phonon calculation",
        )

        spec.input(
            "no_hdf5",
            valid_type=Bool,
            required=False,
            help="write force constants to phonopy yaml, rather than separate HDF5.",
        )
        spec.input(
            "dos",
            valid_type=Bool,
            required=False,
            help="calculate the denity of states",
        )
        spec.input(
            "pdos",
            valid_type=Bool,
            required=False,
            help="calculate the partial denity of states",
        )
        spec.input(
            "bands",
            valid_type=Bool,
            required=False,
            help="calculate the phonon band structure",
        )
        spec.input(
            "symmetrize",
            valid_type=Bool,
            required=False,
            help="symmetrize force constants",
        )
        spec.input(
            "qpoint_file",
            valid_type=Str,
            required=False,
            help="the file for q-points in phonon calculation",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "mlip.ph_parser"

        # Define outputs. The default is a dictionary with the content of the
        # phonon file
        spec.output(
            "results_dict",
            valid_type=Dict,
            help="The `results_dict` output node of the successful calculation.",
        )
        spec.output("xyz_output", valid_type=SinglefileData)
        spec.output("force_constant", valid_type=SinglefileData)
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
        xyz_filename = (self.inputs.out).value

        # Gather node inputs for use with janus CLI
        supercell = (self.inputs.supercell).value
        fmax = (self.inputs.fmax).value
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

        # option to minimize the unit cell before phonons
        minimize = (self.inputs.minimize).value
        if minimize:
            codeinfo.cmdline_params += [
                "--minimize",
            ]
            codeinfo.cmdline_params += [
                "--fmax",
                fmax,
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

        # properties left in just in case for further expansion
        if "properties" in self.inputs:
            properties = self.inputs.properties.value
            codeinfo.cmdline_params += ["--properties", properties]

        calcinfo.retrieve_list.append(xyz_filename)
        calcinfo.retrieve_list.append("aiida-force_constants.hdf5")
        calcinfo.retrieve_list.append("aiida-dos.dat")
        calcinfo.retrieve_list.append("aiida-pdos.dat")
        calcinfo.retrieve_list.append("aiida-auto_bands.yml.xz")
        # calcinfo.retrieve_list.append("aiida-bands.hdf5")

        return calcinfo
