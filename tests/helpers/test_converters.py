"""Tests for helper converters.py."""

from pathlib import Path

import numpy as np

from aiida.orm import StructureData, TrajectoryData

from aiida_mlip.helpers.converters import convert_numpy, xyz_to_aiida_traj


def test_convert_numpy():
    """Test for the convert_numpy function."""
    input_dict = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    expected_output = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert convert_numpy(input_dict) == expected_output


def test_xyz_to_aiida_traj(structure_folder):
    """Test for the xyz_to_aiida_traj function."""
    last_structure, trajectory = xyz_to_aiida_traj(Path(structure_folder / "traj.xyz"))
    assert isinstance(last_structure, StructureData)
    assert isinstance(trajectory, TrajectoryData)
    assert trajectory.numsteps == 5
