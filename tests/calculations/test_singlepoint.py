"""Tests for singlepoint calculation."""

from ase.build import bulk
import pytest

from aiida.common import datastructures
from aiida.orm import Str, StructureData

from aiida_mlip.data.model import ModelData


def test_singlepoint(
    fixture_sandbox, generate_calc_job, tmp_path, janus_code, file_regression
):
    """Test singlepoint calculation"""
    entry_point_name = "janus.sp"
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace_mp"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "calctype": Str("singlepoint"),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace_mp",
            cache_dir=tmp_path,
        ),
        "device": Str("cpu"),
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    # pylint:disable=line-too-long
    cmdline_params = [
        "singlepoint",
        "--arch",
        "mace_mp",
        "--struct",
        "aiida.cif",
        "--device",
        "cpu",
        "--log",
        "aiida.log",
        "--calc-kwargs",
        f"{{'model_paths': '{tmp_path}/mace_mp/mace_mp_small.model', 'default_dtype': 'float64'}}",
        "--write-kwargs",
        "{'filename': 'aiida-results.xyz'}",
    ]

    retrieve_list = [
        calc_info.uuid,
        "aiida.log",
        "aiida-results.xyz",
        "aiida-stdout.txt",
    ]

    print(sorted(cmdline_params))
    print(sorted(calc_info.codes_info[0].cmdline_params))
    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)

    with fixture_sandbox.open("aiida.cif") as handle:
        input_written = handle.read()
        print(input_written)
    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == ["aiida.cif"]
    file_regression.check(input_written, encoding="utf-8", extension=".cif")


def test_sp_error(fixture_sandbox, generate_calc_job, tmp_path, janus_code):
    """Test singlepoint calculation"""
    entry_point_name = "janus.sp"
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code": janus_code,
        "architecture": Str("mace_mp"),
        "precision": Str("float64"),
        "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
        "calctype": Str("wrong_type"),
        "model": ModelData.download(
            "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
            architecture="mace_mp",
            cache_dir=tmp_path,
        ),
        "device": Str("cpu"),
    }
    with pytest.raises(ValueError):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)
