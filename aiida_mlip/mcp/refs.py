"""Tagged reference resolvers, config synthesis, and error types for MCP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

from aiida.common import exceptions as aiida_exc
from aiida.orm import Code, QueryBuilder, StructureData, load_code, load_node
import ase.io
import yaml

from aiida_mlip.data.config import JanusConfigfile
from aiida_mlip.data.model import ModelData

DEFAULT_CONFIG_CACHE_DIR = Path("~/.cache/aiida-mlip-mcp/configs")
DEFAULT_PRESETS_PATH = Path("~/.config/aiida-mlip-mcp/presets.yml")


class MCPToolError(Exception):
    """
    Structured domain error converted into an MCP error response dictionary.

    Parameters
    ----------
    error : str
        Machine-readable error code (e.g. ``validation_error``).
    message : str
        Human-readable description of the error.
    hint : str | None
        Optional actionable recovery hint.
    """

    def __init__(
        self,
        error: str,
        message: str,
        hint: str | None = None,
    ) -> None:
        """
        Initialize the MCPToolError.

        Parameters
        ----------
        error : str
            Machine-readable error code.
        message : str
            Human-readable description of the error.
        hint : str | None
            Optional actionable recovery hint.
        """
        super().__init__(message)
        self.error = error
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the error into the standard MCP tool error envelope.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``status``, ``error``, ``message``, and ``hint``.
        """
        return {
            "status": "error",
            "error": self.error,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class DryRunModelRef:
    """
    Lightweight placeholder for a ``uri:`` model during ``dry_run=True``.

    Attributes
    ----------
    uri : str
        Remote model URI.
    architecture : str
        MLIP architecture name.
    model_hash : str
        Empty string placeholder matching ``ModelData.model_hash``.
    """

    uri: str
    architecture: str
    model_hash: str = ""


def _parse_tagged_ref(ref: str) -> tuple[str | None, str]:
    """
    Split a tagged reference string into ``(tag, value)``.

    Parameters
    ----------
    ref : str
        Reference string such as ``pk:42``, ``uuid:...``, ``file:...``, or
        ``uri:...``.

    Returns
    -------
    tuple[str | None, str]
        Tuple of lowercase tag (or ``None`` if untagged) and remaining value.
    """
    stripped = ref.strip()
    for prefix in ("pk", "uuid", "file", "uri"):
        token = f"{prefix}:"
        if stripped.startswith(token):
            return prefix, stripped[len(token) :]
    return None, stripped


def _load_typed_node(identifier: int | str, expected_type: type, label: str) -> Any:
    """
    Load an AiiDA node by PK or UUID and verify its ORM type.

    Parameters
    ----------
    identifier : int | str
        Node PK or UUID.
    expected_type : type
        Required ORM class.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    Any
        Loaded AiiDA node of ``expected_type``.

    Raises
    ------
    MCPToolError
        If the node does not exist or has the wrong type.
    """
    try:
        node = load_node(identifier)
    except aiida_exc.NotExistent as exc:
        raise MCPToolError(
            "node_not_found",
            f"{label} node '{identifier}' was not found in the active AiiDA profile.",
        ) from exc
    except aiida_exc.MultipleObjectsError as exc:
        raise MCPToolError(
            "validation_error",
            f"Ambiguous {label} identifier '{identifier}'; provide a longer UUID.",
        ) from exc
    if not isinstance(node, expected_type):
        raise MCPToolError(
            "validation_error",
            f"Node '{identifier}' is {type(node).__name__}, expected "
            f"{expected_type.__name__}.",
        )
    return node


def resolve_structure(ref: str) -> StructureData:
    """
    Resolve a tagged structure reference into an AiiDA ``StructureData``.

    Parameters
    ----------
    ref : str
        Structure reference: ``pk:<int>``, ``uuid:<str>``, ``file:<path>``,
        bare integer PK, or existing file path.

    Returns
    -------
    StructureData
        Resolved AiiDA ``StructureData`` node (unstored when loaded from file).

    Raises
    ------
    MCPToolError
        If the reference is invalid, missing, unreadable, or the wrong type.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise MCPToolError("validation_error", "Structure reference must be non-empty.")

    tag, value = _parse_tagged_ref(ref)
    if tag == "pk" or (tag is None and value.isdigit()):
        try:
            pk = int(value)
        except ValueError as exc:
            raise MCPToolError(
                "validation_error", f"Invalid structure PK: '{value}'."
            ) from exc
        return _load_typed_node(pk, StructureData, "StructureData")

    if tag == "uuid":
        return _load_typed_node(value, StructureData, "StructureData")

    if tag == "file" or (tag is None and Path(value).expanduser().is_file()):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise MCPToolError(
                "validation_error", f"Structure file does not exist: '{path}'."
            )
        try:
            atoms = ase.io.read(path)
        except Exception as exc:
            raise MCPToolError(
                "validation_error",
                f"Failed to read structure file '{path}' with ASE: {exc}",
            ) from exc
        return StructureData(ase=atoms)

    raise MCPToolError(
        "validation_error",
        f"Invalid structure reference '{ref}'. Expected 'pk:<id>', 'uuid:<uuid>', "
        "'file:<path>', or an existing file path.",
    )


def resolve_model(
    ref: str,
    arch: str | None = None,
    *,
    dry_run: bool = False,
) -> ModelData | DryRunModelRef:
    """
    Resolve a tagged model reference into ``ModelData`` or ``DryRunModelRef``.

    Parameters
    ----------
    ref : str
        Model reference: ``pk:<int>``, ``uuid:<str>``, ``file:<path>``, or
        ``uri:<url>``.
    arch : str | None
        MLIP architecture name. Required for ``file:`` and ``uri:`` references;
        validated against ``ModelData.architecture`` if provided for ``pk:``
        or ``uuid:``.
    dry_run : bool
        When ``True``, ``uri:`` references return a ``DryRunModelRef`` without
        performing network downloads or disk writes.

    Returns
    -------
    ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.

    Raises
    ------
    MCPToolError
        If the reference is invalid, missing, or has an architecture mismatch.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise MCPToolError("validation_error", "Model reference must be non-empty.")

    tag, value = _parse_tagged_ref(ref)
    if tag in ("pk", "uuid") or (tag is None and value.isdigit()):
        if tag == "pk" or tag is None:
            try:
                identifier: int | str = int(value)
            except ValueError as exc:
                raise MCPToolError(
                    "validation_error", f"Invalid model PK: '{value}'."
                ) from exc
        else:
            identifier = value
        model = _load_typed_node(identifier, ModelData, "ModelData")
        if arch is not None and arch != model.architecture:
            raise MCPToolError(
                "validation_error",
                f"Architecture mismatch: model node has '{model.architecture}', "
                f"but arch='{arch}' was requested.",
            )
        return model

    if tag == "file" or (tag is None and Path(value).expanduser().is_file()):
        if not arch:
            raise MCPToolError(
                "validation_error",
                "'arch' is required when loading a model from a local file.",
            )
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise MCPToolError(
                "validation_error", f"Model file does not exist: '{path}'."
            )
        return ModelData.from_local(file=path, architecture=arch)

    if tag == "uri" or (tag is None and value.startswith(("http://", "https://"))):
        if not arch:
            raise MCPToolError(
                "validation_error",
                "'arch' is required when loading a model from a URI.",
            )
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise MCPToolError(
                "validation_error",
                f"Invalid model URI '{value}'. Expected an http:// or https:// URL.",
            )
        if dry_run:
            return DryRunModelRef(uri=value, architecture=arch)
        return ModelData.from_uri(uri=value, architecture=arch)

    raise MCPToolError(
        "validation_error",
        f"Invalid model reference '{ref}'. Expected 'pk:<id>', 'uuid:<uuid>', "
        "'file:<path>', or 'uri:<url>'.",
    )


def resolve_code(code: str) -> Code:
    """
    Resolve an AiiDA ``Code`` by label or full label.

    Parameters
    ----------
    code : str
        Code label (e.g. ``janus@localhost``).

    Returns
    -------
    Code
        Loaded AiiDA ``Code`` node.

    Raises
    ------
    MCPToolError
        With ``error="code_not_found"`` if missing or ambiguous.
    """
    try:
        return load_code(code)
    except (aiida_exc.NotExistent, aiida_exc.MultipleObjectsError) as exc:
        qb = QueryBuilder().append(Code)
        available = [c.full_label for (c,) in qb.all()[:10]]
        hint = (
            f"Available codes: {', '.join(available)}"
            if available
            else "No AiiDA codes are registered. Register one with 'verdi code create'."
        )
        raise MCPToolError(
            "code_not_found",
            f"Could not resolve AiiDA code '{code}': {exc}",
            hint=hint,
        ) from exc


def build_config_node(
    overrides: dict[str, Any] | None,
    *,
    force_no_tracker: bool = True,
    cache_dir: Path | None = None,
) -> JanusConfigfile:
    """
    Synthesize a persistent ``JanusConfigfile`` node from a dictionary.

    Writes the YAML file to a persistent cache directory so that
    ``JanusConfigfile.read_yaml()`` (which opens ``self.filepath`` on disk)
    always succeeds across submission and parsing.

    Parameters
    ----------
    overrides : dict[str, Any] | None
        Configuration key-value pairs to serialize into YAML.
    force_no_tracker : bool
        When ``True``, sets ``tracker: false`` unless explicitly specified in
        ``overrides``. Must be ``False`` for ``mlip.train`` configs.
    cache_dir : Path | None
        Directory in which to store synthesized YAML files.

    Returns
    -------
    JanusConfigfile
        Unstored ``JanusConfigfile`` pointing to the persistent YAML file.
    """
    cfg = dict(overrides or {})
    if force_no_tracker:
        cfg.setdefault("tracker", False)

    rendered = yaml.safe_dump(cfg, sort_keys=True)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
    target_dir = (cache_dir or DEFAULT_CONFIG_CACHE_DIR).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = target_dir / f"config_{digest}.yml"
    if not yaml_path.exists():
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".config_{digest}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            tmp_path = Path(handle.name)
        os.replace(tmp_path, yaml_path)
    return JanusConfigfile(file=yaml_path)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries without mutating inputs.

    Parameters
    ----------
    base : dict[str, Any]
        Base dictionary.
    patch : dict[str, Any]
        Override dictionary taking precedence over ``base``.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_metadata_options(
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA ``metadata.options`` from optional named presets and overrides.

    Always ensures ``resources`` defaults to ``{"num_machines": 1}`` so
    localhost and direct-scheduler submissions succeed out of the box.

    Parameters
    ----------
    hpc_preset : str | None
        Optional preset name defined in the presets YAML file.
    hpc_overrides : dict[str, Any] | None
        Explicit ``metadata.options`` overrides merged on top of the preset.
    presets_path : Path | str | None
        Optional path to the presets YAML file. Falls back to
        ``$AIIDA_MLIP_MCP_PRESETS`` or ``~/.config/aiida-mlip-mcp/presets.yml``.

    Returns
    -------
    dict[str, Any]
        Resolved ``metadata.options`` dictionary.

    Raises
    ------
    MCPToolError
        If ``hpc_preset`` is requested but the file is missing, invalid YAML,
        or does not contain the named preset.
    """
    options: dict[str, Any] = {}
    if hpc_preset is not None:
        env_path = os.environ.get("AIIDA_MLIP_MCP_PRESETS")
        resolved_path = (
            Path(presets_path)
            if presets_path is not None
            else Path(env_path)
            if env_path
            else DEFAULT_PRESETS_PATH
        ).expanduser()
        if not resolved_path.is_file():
            raise MCPToolError(
                "validation_error",
                f"HPC presets file not found: '{resolved_path}'.",
                hint="Create the presets YAML file or pass hpc_overrides directly.",
            )
        try:
            presets = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MCPToolError(
                "validation_error",
                f"Failed to parse HPC presets file '{resolved_path}': {exc}",
            ) from exc
        if not isinstance(presets, dict) or hpc_preset not in presets:
            known = sorted(presets.keys()) if isinstance(presets, dict) else []
            raise MCPToolError(
                "validation_error",
                f"Unknown hpc_preset '{hpc_preset}'.",
                hint=f"Available presets: {', '.join(known)}" if known else None,
            )
        preset_val = presets[hpc_preset]
        if not isinstance(preset_val, dict):
            raise MCPToolError(
                "validation_error",
                f"Preset '{hpc_preset}' must map to a dictionary of metadata.options.",
            )
        options = dict(preset_val)

    if hpc_overrides:
        options = _deep_merge(options, hpc_overrides)

    options.setdefault("resources", {"num_machines": 1})
    return options
