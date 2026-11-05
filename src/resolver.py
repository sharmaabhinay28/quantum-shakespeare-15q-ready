"""Deterministic checkpoint resolution and content hash verification.

Enforces strict local resolution:
- All checkpoint references must be relative paths.
- Absolute paths and path traversal ('..') are strictly rejected.
- When an explicit ckpt_dir is passed, it is the sole authority with zero fallback
  to cwd, repo default, or external paths.
- Default anchor is strictly the repository checkpoints directory.
- Controlled test fixtures are supported via explicit fixture directories.
- Content hash verification raises ValueError immediately upon mismatch without fallback.
"""

import os
import hashlib
import numpy as np


def sha256_of_array(arr: np.ndarray) -> str:
    """Computes SHA-256 of the raw memory buffer of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def resolve_checkpoint_path(
    checkpoint_ref,
    expected_sha256: str = None,
    ckpt_dir: str = None
) -> tuple[np.ndarray, str]:
    """Resolves a checkpoint path portably and deterministically.

    Args:
        checkpoint_ref: Dict (e.g. selection record) or relative filepath string.
        expected_sha256: Expected SHA-256 hex digest of array bytes (optional).
        ckpt_dir: Explicit checkpoint directory. If provided, this directory is the
            sole authority.

    Returns:
        (params_array, resolved_path)

    Raises:
        ValueError: If path is absolute, contains '..', or has content hash mismatch.
        FileNotFoundError: If checkpoint cannot be found in the authoritative directory.
    """
    if isinstance(checkpoint_ref, dict):
        raw_path = checkpoint_ref.get("checkpoint_path", "")
        if expected_sha256 is None:
            expected_sha256 = checkpoint_ref.get("checkpoint_sha256")
    else:
        raw_path = str(checkpoint_ref)

    if not raw_path:
        raise ValueError("Empty checkpoint reference cannot be resolved.")

    # Reject absolute paths
    if os.path.isabs(raw_path):
        raise ValueError(
            f"Absolute checkpoint paths are rejected: {raw_path}. References must be relative."
        )

    # Reject path traversal
    normalized_parts = raw_path.replace(chr(92), "/").split("/")
    if ".." in normalized_parts:
        raise ValueError(
            f"Path traversal ('..') is strictly rejected: {raw_path}"
        )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if ckpt_dir is not None:
        # Explicit ckpt_dir is the SOLE authority
        sole_dir = os.path.abspath(ckpt_dir)
        basename = os.path.basename(raw_path)
        candidate1 = os.path.join(sole_dir, raw_path)
        candidate2 = os.path.join(sole_dir, basename)

        if os.path.isfile(candidate1):
            resolved_path = candidate1
        elif os.path.isfile(candidate2):
            resolved_path = candidate2
        else:
            raise FileNotFoundError(
                f"Checkpoint '{raw_path}' not found in authoritative ckpt_dir: {sole_dir}. "
                f"External fallbacks and cwd searching are strictly prohibited."
            )
    else:
        # Default anchor is repo checkpoints/ directory
        default_dir = os.path.join(repo_root, "checkpoints")
        basename = os.path.basename(raw_path)
        cand_default = os.path.join(default_dir, basename)
        cand_rel = os.path.abspath(os.path.join(repo_root, raw_path))

        if os.path.isfile(cand_default):
            resolved_path = cand_default
        else:
            raise FileNotFoundError(
                f"Checkpoint '{raw_path}' not found in default repo checkpoints: {default_dir}. "
                f"External fallbacks and cwd searching are strictly prohibited."
            )

    authority = os.path.realpath(ckpt_dir if ckpt_dir is not None else default_dir)
    resolved_path = os.path.realpath(resolved_path)
    if os.path.commonpath([authority, resolved_path]) != authority:
        raise ValueError("Checkpoint resolves outside its authoritative directory.")
    arr = np.load(resolved_path, allow_pickle=False)

    if expected_sha256 is not None:
        actual_sha = sha256_of_array(arr)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"Content hash mismatch for {resolved_path}!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha}"
            )

    return arr, resolved_path
