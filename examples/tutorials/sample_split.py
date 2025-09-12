"""Split descriptors files into Test, Train and Validate."""

from __future__ import annotations

from pathlib import Path

from aiida.orm import Dict, Str
from aiida_workgraph import task
from ase.io import read, write
from fpsample import fps_npdu_kdtree_sampling as sample
import numpy as np


def extract(data, inds):
    """Extract elements from a list based on indices."""
    return [data[i] for i in inds]


def sampling(data, dims, n_samples):
    """Perform furthest point sampling on the given data."""
    n = len(data)
    if n == 0:
        return set()
    ldata = np.array(data)
    ldata.reshape(n, dims)
    # Ensure n_samples does not exceed the number of available data points
    actual_n_samples = min(n_samples, n)
    if actual_n_samples == 0:
        return set()
    return set(sample(ldata, n_samples=actual_n_samples, start_idx=0))


def write_samples(frames, inds, f_s):
    """Write selected frames to a file."""
    for i in inds:
        write(f_s, frames[i], write_info=True, append=True)


@task.calcfunction()
def process_and_split_data(
    config_types: str,
    n_samples: int,
    prefix: str,
    scale: float,
    append_mode: bool,
    trajectory_path: str | None = None,
    **trajectory_data,
) -> Dict:
    """
    Split a trajectory into training, validation, and test sets.

    Parameters
    ----------
    config_types: str
        List of configuration types to process.
    n_samples:
        int The target number of samples for each configuration type.
    scale: float
        Scaling factor for the MACE descriptors.
    prefix str:
        A prefix string for the output filenames.
    append_mode: bool
        If True, append to existing files. Otherwise, overwrite.
    trajectory_path: str (Optional)
        Path to trajectory

    Returns
    -------
    aiida.orm.nodes.data.dict.Dict
        An instance of a orm.Dict with file names and file paths

    """
    if (trajectory_path is None) == (trajectory_data is None):
        raise ValueError("Please specify the trajectory_path or trajectory_data.")

    config_types = config_types.value
    n_samples = n_samples.value
    prefix = prefix.value
    scale = scale.value
    append_mode = append_mode.value

    if trajectory_path is None:
        print("Using trajectory from data: trajectory_data")

        atoms = []
        for _, data in trajectory_data.items():
            # Each trajectory_data item is a single structure
            with data.open() as handle:
                ase_atoms = read(handle, format="extxyz")
            atoms.append(ase_atoms)

    else:
        print(f"Using trajectory from path: {trajectory_path}")
        traj_path = Path(trajectory_path)
        if not traj_path.exists():
            print(f"Error: Trajectory file not found at {traj_path}")
            return None
        atoms = read(traj_path, index=":")

    if prefix and not prefix.endswith("-"):
        prefix += "-"

    train_file = Path(f"{prefix}train.xyz")
    valid_file = Path(f"{prefix}valid.xyz")
    test_file = Path(f"{prefix}test.xyz")

    if not append_mode:
        _ = [p.unlink(missing_ok=True) for p in [train_file, valid_file, test_file]]

    print(f"create files: {train_file=}, {valid_file=} and {test_file=}")

    stats = {}
    for i, struct in enumerate(atoms):
        system_name = struct.info.get("system_name", "unknown_system")
        config_type = struct.info.get("config_type", "all")
        key = (config_type, system_name)
        if key not in stats:
            stats[key] = []
        stats[key].append(i)

    k = 0
    for key, indices in stats.items():
        run_type, system = key

        if run_type in config_types or "all" in config_types or run_type == "all":
            n = len(indices)
            print(f"Processing: {key}, {n} frames")

            if n >= n_samples:
                ns_train_target = int(0.8 * n_samples)
                ns_total_target = n_samples
            else:
                ns_train_target = int(0.8 * n)
                ns_total_target = n

            specs = set(atoms[indices[0]].get_chemical_symbols())
            dims = len(specs)

            desc_per_spec = [
                [atoms[x].info[f"mace_mp_{s}_descriptor"] * scale for s in specs]
                for x in indices
            ]

            ind_spec_train = sampling(desc_per_spec, dims, ns_train_target)
            train_ind = extract(indices, list(ind_spec_train))

            ns_train_actual = len(train_ind)

            if (
                key[0] == "geometry_optimisation"
                and indices
                and indices[-1] not in train_ind
            ):
                train_ind.append(indices[-1])

            left_indices = list(set(indices) - set(train_ind))

            nvt_target = ns_total_target - ns_train_actual
            if nvt_target < 0:
                nvt_target = 0
                k += abs(ns_total_target - ns_train_actual)

            print(
                f"  {key}: total={n}, train_target={ns_train_target}, \
                    vt_target={nvt_target}"
            )

            if left_indices and nvt_target > 0:
                desc_per_spec_vt = [
                    [atoms[x].info[f"mace_mp_{s}_descriptor"] * scale for s in specs]
                    for x in left_indices
                ]
                vt_spec = sampling(desc_per_spec_vt, dims, nvt_target)
                vt_ind = extract(left_indices, list(vt_spec))

                test_ind = vt_ind[0::2]
                valid_ind = vt_ind[1::2]
            else:
                test_ind, valid_ind = [], []

            write_samples(atoms, train_ind, train_file)
            write_samples(atoms, test_ind, test_file)
            write_samples(atoms, valid_ind, valid_file)
            return Dict(
                {
                    "train_file": str(Path(train_file).resolve()),
                    "test_file": str(Path(test_file).resolve()),
                    "valid_file": str(Path(valid_file).resolve()),
                }
            )

        print(
            f"Config type '{run_type}' not in target list. \
                    Adding all {len(indices)} frames to training set."
        )
        write_samples(atoms, indices, train_file)

        return Dict(
            {
                "train_file": str(Path(train_file).resolve()),
            }
        )

    return Str(f"Found {k} structures that were too similar during sampling.")
