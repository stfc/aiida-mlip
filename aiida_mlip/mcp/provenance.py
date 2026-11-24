"""Canonical intent hashing and AiiDA provenance deduplication helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from aiida.orm import ProcessNode, QueryBuilder, StructureData

from aiida_mlip.data.model import ModelData
from aiida_mlip.mcp.refs import DryRunModelRef

INTENT_HASH_EXTRA = "aiida_mlip_mcp_intent_hash"
ACTIVE_PROCESS_STATES = ("created", "waiting", "running")
FAILED_PROCESS_STATES = ("excepted", "killed")


def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file on disk.

    Parameters
    ----------
    path : Path
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def structure_fingerprint(struct: StructureData) -> dict[str, Any]:
    """
    Compute a geometry-based fingerprint for an AiiDA ``StructureData``.

    Because the fingerprint depends only on atomic symbols, periodic boundary
    conditions, cell matrix, and atomic positions (rounded to 8 decimal
    places), identical structures loaded via ``file:`` and ``pk:`` hash to the
    same value.

    Parameters
    ----------
    struct : StructureData
        Input AiiDA structure node (stored or unstored).

    Returns
    -------
    dict[str, Any]
        JSON-serializable geometry fingerprint dictionary.
    """
    atoms = struct.get_ase()
    return {
        "symbols": atoms.get_chemical_symbols(),
        "pbc": [bool(flag) for flag in atoms.get_pbc()],
        "cell": [
            [round(float(val), 8) for val in row] for row in atoms.get_cell().tolist()
        ],
        "positions": [
            [round(float(val), 8) for val in row]
            for row in atoms.get_positions().tolist()
        ],
    }


def model_fingerprint(model: ModelData | DryRunModelRef) -> dict[str, Any]:
    """
    Compute a canonical fingerprint for a ``ModelData`` or ``DryRunModelRef``.

    Parameters
    ----------
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run URI reference.

    Returns
    -------
    dict[str, Any]
        JSON-serializable model fingerprint dictionary.
    """
    if isinstance(model, DryRunModelRef):
        return {"architecture": model.architecture, "uri": model.uri}
    return {"architecture": model.architecture, "model_hash": model.model_hash}


def compute_intent_hash(entry_point: str, payload: dict[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 intent hash for a calculation or workflow.

    Parameters
    ----------
    entry_point : str
        Normalized calculation or workflow identifier (e.g. ``mlip.opt``).
    payload : dict[str, Any]
        Canonical JSON-serializable input payload.

    Returns
    -------
    str
        Hexadecimal SHA-256 intent hash.
    """
    serialized = json.dumps(
        {"entry_point": entry_point, "payload": payload},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def find_matching_process(
    intent_hash: str,
    *,
    dedupe: Literal["reuse", "strict", "force"] = "reuse",
) -> ProcessNode | None:
    """
    Search AiiDA provenance for an existing process with the same intent hash.

    Parameters
    ----------
    intent_hash : str
        Canonical SHA-256 intent hash computed by ``compute_intent_hash``.
    dedupe : Literal["reuse", "strict", "force"]
        Deduplication mode:
        - ``"reuse"``: matches newest active OR cleanly finished process.
        - ``"strict"``: matches newest active (in-flight) process only.
        - ``"force"``: disables lookup and always returns ``None``.

    Returns
    -------
    ProcessNode | None
        Most recent matching ``ProcessNode``, or ``None`` if no match exists.
    """
    if dedupe == "force":
        return None

    state_filter: dict[str, Any]
    if dedupe == "strict":
        state_filter = {"attributes.process_state": {"in": list(ACTIVE_PROCESS_STATES)}}
    else:
        state_filter = {
            "or": [
                {"attributes.process_state": {"in": list(ACTIVE_PROCESS_STATES)}},
                {
                    "and": [
                        {"attributes.process_state": "finished"},
                        {"attributes.exit_status": 0},
                    ]
                },
            ]
        }

    qb = QueryBuilder()
    qb.append(
        ProcessNode,
        filters={
            f"extras.{INTENT_HASH_EXTRA}": intent_hash,
            **state_filter,
        },
        tag="proc",
    )
    qb.order_by({"proc": {"ctime": "desc"}})
    result = qb.first()
    return result[0] if result is not None else None


def tag_process_intent(node: ProcessNode, intent_hash: str) -> None:
    """
    Store the canonical intent hash as an AiiDA extra on a submitted process.

    Parameters
    ----------
    node : ProcessNode
        Submitted AiiDA process node.
    intent_hash : str
        Canonical SHA-256 intent hash.
    """
    node.base.extras.set(INTENT_HASH_EXTRA, intent_hash)


def count_failed_attempts(intent_hash: str | None) -> int:
    """
    Count prior failed, excepted, or killed processes for an intent hash.

    Parameters
    ----------
    intent_hash : str | None
        Canonical intent hash, or ``None`` if the node is untagged.

    Returns
    -------
    int
        Number of failed attempts recorded in the active AiiDA profile.
    """
    if not intent_hash:
        return 0

    qb = QueryBuilder()
    qb.append(
        ProcessNode,
        filters={
            f"extras.{INTENT_HASH_EXTRA}": intent_hash,
            "or": [
                {"attributes.process_state": {"in": list(FAILED_PROCESS_STATES)}},
                {
                    "and": [
                        {"attributes.process_state": "finished"},
                        {"attributes.exit_status": {"!==": 0}},
                    ]
                },
            ],
        },
    )
    return qb.count()
