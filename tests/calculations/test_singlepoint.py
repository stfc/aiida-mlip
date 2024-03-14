"""Tests for singlepoint calculation."""

from ase.build import bulk
import pytest

from aiida.common import datastructures
from aiida.orm import Str, StructureData
from aiida.common import NotExistent
from aiida.engine import run
from aiida.orm import Code, Str, StructureData, load_node, load_code
from aiida.plugins import CalculationFactory

from aiida_mlip.data.model import ModelData


    # def test_singlepoint(fixture_sandbox, generate_calc_job, tmp_path, janus_code):
    # """Test singlepoint calculation"""
    # entry_point_name = "janus.sp"
    # # pylint:disable=line-too-long
    # inputs = {
    #     "metadata": {"options": {"resources": {"num_machines": 1}}},
    #     "code": janus_code,
    #     "architecture": Str("mace_mp"),
    #     "precision": Str("float64"),
    #     "structure": StructureData(ase=bulk("NaCl", "rocksalt", 5.63)),
    #     "calctype": Str("singlepoint"),
    #     "model": ModelData.download(
    #         "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
    #         architecture="mace_mp",
    #         cache_dir=tmp_path,
    #     ),
    #     "device": Str("cpu"),
    # }

    # calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    # # pylint:disable=line-too-long
    # cmdline_params = [
    #     "singlepoint",
    #     "--arch",
    #     "mace_mp",
    #     "--struct",
    #     "aiida.cif",
    #     "--device",
    #     "cpu",
    #     "--log",
    #     "aiida.log",
    #     "--calc-kwargs",
    #     f"{{'model_paths': '{tmp_path}/mace_mp/mace_mp_small.model', 'default_dtype': 'float64'}}",
    #     "--write-kwargs",
    #     "{'filename': 'aiida-results.xyz'}",
    # ]

    # retrieve_list = [
    #     calc_info.uuid,
    #     "aiida.log",
    #     "aiida-results.xyz",
    #     "aiida-stdout.txt",
    # ]

    # print(sorted(cmdline_params))
    # print(sorted(calc_info.codes_info[0].cmdline_params))
    # # Check the attributes of the returned `CalcInfo`
    # assert isinstance(calc_info, datastructures.CalcInfo)
    # assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    # assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    # assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    # assert sorted(fixture_sandbox.get_content_list()) == ["aiida.cif"]


def test_sp_error(fixture_sandbox, generate_calc_job, tmp_path, janus_code):
    """Test singlepoint calculation with error input"""
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

def test_run_sp(tmp_path, janus_code):
    """Test singlepoint calculation"""
    entry_point_name = "janus.sp"
    # pylint:disable=line-too-long
    inputs = {
        "metadata": {"options": {"resources": {"num_machines": 1}}},
        "code":  janus_code,
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
    Singlepointcalc = CalculationFactory("janus.sp")
    result = run(Singlepointcalc, **inputs)

    expected_res = {
        'pbc': [True, True, True],
        'cell': [[3.9810111780803, 0.0, 0.0],
        [1.9905055890401, 3.4476568129673, 0.0],
        [1.9905055890401, 1.1492189376558, 3.2504820155376]],
        'info': {'energy': -6.7575203839729,
        'stress': [[-0.005816546985101, 1.0729600140092e-18, 2.0594053733835e-19],
        [1.0729600140092e-18, -0.005816546985101, 5.2906717239225e-18],
        [2.0594053733835e-19, 5.2906717239225e-18, -0.005816546985101]],
        'unit_cell': 'conventional',
        'spacegroup': 'P 1',
        'free_energy': -6.7575203839729},
        'forces': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'numbers': [11, 17],
        'positions': [[0.0, 0.0, 0.0], [3.98101118, 2.29843788, 1.62524101]],
        'spacegroup_kinds': [0, 1]
        }
    
    assert 'results_dict' in result
    obtained_res = result["results_dict"].get_dict()
    assert 'log_output' in result
    assert 'xyz_output' in result
    assert 'std_output' in result
    assert expected_res == obtained_res