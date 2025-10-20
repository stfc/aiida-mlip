"""Split descriptors files into Test, Train and Validate."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, SupportsIndex, TextIO, TypeVar

from ase.io import read, write
from fpsample import fps_npdu_kdtree_sampling as sample
import numpy as np

T = TypeVar("T")


def extract(data: Sequence[T], inds: Iterable[SupportsIndex]) -> list[T]:
    """Extract elements from a list based on indices."""
    return [data[i] for i in inds]


def sampling(data: Sequence[T], dims: int, n_samples: int) -> set[T]:
    """Perform furthest point sampling on the given data."""
    n = len(data)
    if not n:
        return set()
    ldata = np.array(data)
    ldata.reshape(n, dims)
    # Ensure n_samples does not exceed the number of available data points
    actual_n_samples = min(n_samples, n)
    if not actual_n_samples:
        return set()
    return set(sample(ldata, n_samples=actual_n_samples, start_idx=0))


def write_samples(
    frames: Sequence[Any], inds: Iterable[SupportsIndex], f_s: TextIO
) -> None:
    """Write selected frames to a file."""
    for i in inds:
        write(f_s, frames[i], write_info=True, append=True)


def process_and_split_data(
    config_types: str,
    n_samples: int,
    prefix: str,
    scale: float,
    append_mode: bool,
    **trajectory_data,
) -> dict:
    """
    Split a trajectory into training, validation, and test sets.

    Parameters
    ----------
        trajectory_data : dict
            A dict of structures as SinglefileData type.
        config_types : list
            List of configuration types to process.
        n_samples : int
            The target number of samples for each configuration type.
        scale : float
            Scaling factor for the MACE descriptors.
        prefix : str
            A prefix string for the output filenames.
        append_mode : bool
            If True, append to existing files. Otherwise, overwrite.

    Returns
    -------
        files : dict
            A dict instance with file paths.
    """
    if isinstance(trajectory_data, dict):
        atoms = []
        for data in trajectory_data.values():
            with data.open() as handle:
                ase_atoms = read(handle, format="extxyz")
            atoms.append(ase_atoms)

    else:
        traj_path = Path(trajectory_data)
        if not traj_path.exists():
            raise FileNotFoundError(f"Error: Trajectory file not found at {traj_path}")
        atoms = read(traj_path, index=":")

    if prefix:
        prefix = prefix.rstrip("-") + "-"

    train_file = Path(f"{prefix}train.xyz")
    valid_file = Path(f"{prefix}valid.xyz")
    test_file = Path(f"{prefix}test.xyz")

    if not append_mode:
        for file in (train_file, valid_file, test_file):
            file.unlink(missing_ok=True)

    print(f"create files: {train_file=}, {valid_file=} and {test_file=}")

    stats = {}
    for i, structs in enumerate(atoms):
        system_name = structs.info.get("system_name", "unknown_system")
        config_type = structs.info.get("config_type", "all")
        key = (config_type, system_name)
        stats.setdefault(key, [])
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
            atoms_num = len(specs)

            desc_per_spec = [
                [atoms[x].info[f"mace_mp_{s}_descriptor"] * scale for s in specs]
                for x in indices
            ]

            ind_spec_train = sampling(desc_per_spec, atoms_num, ns_train_target)
            train_ind = extract(indices, ind_spec_train)

            ns_train_actual = len(train_ind)

            if (
                key[0] == "geometry_optimisation"
                and indices
                and indices[-1] not in train_ind
            ):
                train_ind.append(indices[-1])

            leftover_indices = list(set(indices) - set(train_ind))

            nvt_target = ns_total_target - ns_train_actual
            if nvt_target < 0:
                nvt_target = 0
                k += abs(ns_total_target - ns_train_actual)

            print(
                f"  {key}: total={n}, train_target={ns_train_target}, \
                    vt_target={nvt_target}"
            )

            if leftover_indices and nvt_target > 0:
                desc_per_spec_vt = [
                    [atoms[x].info[f"mace_mp_{s}_descriptor"] * scale for s in specs]
                    for x in leftover_indices
                ]
                vt_spec = sampling(desc_per_spec_vt, atoms_num, nvt_target)
                vt_ind = extract(leftover_indices, vt_spec)

                test_ind = vt_ind[0::2]
                valid_ind = vt_ind[1::2]
            else:
                test_ind, valid_ind = [], []

            write_samples(atoms, train_ind, train_file)
            write_samples(atoms, test_ind, test_file)
            write_samples(atoms, valid_ind, valid_file)
            return {
                "train_file": str(Path(train_file).resolve()),
                "test_file": str(Path(test_file).resolve()),
                "valid_file": str(Path(valid_file).resolve()),
            }

        print(
            f"Config type '{run_type}' not in target list. \
                    Adding all {len(indices)} frames to training set."
        )
        write_samples(atoms, indices, train_file)

        return {
            "train_file": str(Path(train_file).resolve()),
        }

    return f"Found {k} structures that were too similar during sampling."
