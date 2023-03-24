import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, TypedDict, Sequence

from gaitmap_bench._config import MAIN_REPO_ROOT
from gaitmap_challenges import save_run as challenge_save_run
from gaitmap_challenges.challenge_base import BaseChallenge


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
    citations: Sequence[str]
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
        "citations",
    )

    for field in str_fields:
        if not field in value:
            raise ValueError(f"{field} of custom_metadata is required.")
        if not isinstance(value[field], str):
            raise TypeError(f"{field} of custom_metadata must be a string.")

    for field in seq_fields:
        if not field in value:
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
    # TODO: Get this value from the config
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
                    "considered as an official run."
                )
                debug = True
        try:
            _validate_custom_metadata(custom_metadata)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Your custom metadata is not valid.\n{e.message}\n"
                "This is not allowed for non-debug runs. "
                "We will still store your results to ensure that you don't loose anything, but they will be "
                "stored as a debug run. "
                "Check the results and rerun with the correct parameters to ensure that your results are "
                "considered as an official run."
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
        git_dirty_ignore=("results",),
        debug_folder_prefix="_",
    )
