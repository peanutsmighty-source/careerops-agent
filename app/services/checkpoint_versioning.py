from __future__ import annotations


UNVERSIONED_CHECKPOINT = "unversioned"


class CheckpointVersionError(ValueError):
    pass


def checkpoint_version(state: dict) -> str:
    value = state.get("graph_version")
    return str(value) if value else UNVERSIONED_CHECKPOINT


def require_checkpoint_version(
    state: dict, *, graph_name: str, expected_version: str
) -> None:
    actual = checkpoint_version(state)
    if actual != expected_version:
        raise CheckpointVersionError(
            f"{graph_name} checkpoint version '{actual}' is incompatible with "
            f"runtime version '{expected_version}'; migrate the checkpoint explicitly"
        )


def require_explicit_source_version(state: dict, *, source_version: str) -> None:
    actual = checkpoint_version(state)
    if actual != source_version:
        raise CheckpointVersionError(
            f"checkpoint source version is '{actual}', not requested '{source_version}'"
        )
