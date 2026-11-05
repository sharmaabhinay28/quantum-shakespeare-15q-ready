"""Guard utilities to protect frozen research artifacts from accidental modification.

All mutating entry points must validate destination directories before executing
any computationally expensive operations. Writes into frozen artifacts
(results/, checkpoints/, data/) or existing non-empty run directories are refused.
"""

import os


def get_protected_dirs(repo_root: str = None) -> list[str]:
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return [
        os.path.realpath(os.path.join(repo_root, "results")),
        os.path.realpath(os.path.join(repo_root, "checkpoints")),
        os.path.realpath(os.path.join(repo_root, "data")),
    ]


def guard_output_dir(target_dir: str, repo_root: str = None, allow_existing: bool = False) -> str:
    """Validates that target_dir is not in a protected artifact directory and is safe to write.

    Args:
        target_dir: Directory path where new artifacts will be written.
        repo_root: Root directory of the repository checkout.
        allow_existing: If False, target_dir must not already exist (or must be empty).

    Returns:
        Canonical absolute path to target_dir.

    Raises:
        ValueError: If target_dir is empty, inside protected directories, or already exists.
    """
    if not target_dir:
        raise ValueError("An explicit output directory must be specified.")

    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    target_abs = os.path.realpath(target_dir)
    protected = get_protected_dirs(repo_root)

    for p in protected:
        if target_abs == p or target_abs.startswith(p + os.sep):
            raise ValueError(
                f"Write rejected! Destination '{target_abs}' is inside protected directory '{p}'. "
                f"Overwriting frozen publication artifacts is strictly prohibited."
            )

    if not allow_existing and os.path.exists(target_abs):
        if os.path.isdir(target_abs) and os.listdir(target_abs):
            raise ValueError(
                f"Write rejected! Destination '{target_abs}' already exists and is non-empty. "
                f"A new, non-existent run directory is required."
            )
        elif os.path.isfile(target_abs):
            raise ValueError(
                f"Write rejected! Destination '{target_abs}' is an existing file."
            )

    return target_abs
