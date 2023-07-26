import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TypedDict, Union

from gaitmap_challenges import save_run as challenge_save_run
from gaitmap_challenges.challenge_base import BaseChallenge
from gaitmap_challenges.config import is_debug_run

from gaitmap_bench._config import MAIN_REPO_ROOT

EXPECTED_DEFAULT_SAVE_CONFIG = {
    "path": None,
    "stored_filenames_relative_to": MAIN_REPO_ROOT,
    "use_git": True,
}


class CustomMetadata(TypedDict):
    short_description: str
    long_description: str
    code_authors: Sequence[str]
    algorithm_authors: Sequence[str]
    references: Sequence[str]
    implementation_url: str


def _validate_custom_metadata(value: Dict[str, Any]):
    """Test if the value is a valid CustomMetadata object."""
    str_fields = (
        "short_description",
        "long_description",
        "implementation_url",
    )

    seq_fields = (
        "code_authors",
        "algorithm_authors",
        "references",
    )

    for field in str_fields:
        if field not in value:
            raise ValueError(f"{field} of custom_metadata is required.")
        if not isinstance(value[field], str):
            raise TypeError(f"{field} of custom_metadata must be a string.")

    for field in seq_fields:
        if field not in value:
            raise ValueError(f"{field} of custom_metadata is required.")
        if not isinstance(value[field], (list, tuple)):
            raise TypeError(f"{field} of custom_metadata must be a list or a tuple of strings.")
        for item in value[field]:
            if not isinstance(item, str):
                raise TypeError(f"{field} of custom_metadata must be a sequence of strings.")


def save_run(
    challenge: BaseChallenge,
    entry_name: Union[str, Tuple[str, ...]],
    *,
    custom_metadata: CustomMetadata,
    path: Optional[Union[str, Path]] = None,
    stored_filenames_relative_to: Union[str, Path] = MAIN_REPO_ROOT,
    use_git: bool = True,
):
    debug = is_debug_run()
    try:
        _validate_custom_metadata(custom_metadata)
    except (TypeError, ValueError) as e:
        if debug is True:
            warnings.warn(
                "Your custom metadata is not valid. "
                "This is fine for debug-runs, but you should fix this before performing an official run. "
                f"\n\nError: {e}",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Your custom metadata is not valid. "
                "This is not allowed for non-debug runs. "
                "We will still store your results to ensure that you don't loose anything, but they will be "
                "stored as a debug run. "
                "Check the results and rerun with the correct parameters to ensure that your results are "
                "considered as an official run."
                f"\n\nError: {e}",
                stacklevel=2,
            )
            debug = True
    if debug is False:
        # We make sure that a couple of parameters are set to the expected values
        for key, value in EXPECTED_DEFAULT_SAVE_CONFIG.items():
            if locals()[key] != value:
                warnings.warn(
                    f"Expected the value of {key} to be {value} but it was {locals()[key]}. "
                    "This is not allowed for non-debug runs. "
                    "We will still store your results to ensure that you don't loose anything, but they will be "
                    "stored as a debug run. "
                    "Check the results and rerun with the correct parameters to ensure that your results are "
                    "considered as an official run.",
                    stacklevel=2,
                )
                debug = True

    return challenge_save_run(
        challenge=challenge,
        entry_name=entry_name,
        custom_metadata=custom_metadata,
        debug_run=debug,
        path=path,
        stored_filenames_relative_to=stored_filenames_relative_to,
        use_git=use_git,
        git_dirty_ignore=("results", "docs", ".run_configs", ".github", "gaitmap_paper"),
        debug_folder_prefix="_",
        _force_local_debug_run_setting=True,
        _caller_file_path_stack_offset=1,
    )
