from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from gaitmap_challenges import save_run as challenge_save_run
from gaitmap_challenges.challenge_base import BaseChallenge


def save_run(
    challenge: BaseChallenge,
    entry_name: Union[str, Tuple[str, ...]],
    *,
    custom_metadata: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
    stored_filenames_relative_to: Optional[Union[str, Path]] = None,
    use_git: bool = True,
):
    # TODO: For now this just passes all information through.
    #  In the future, this will chaeck the config and overwrite certain values e.g. the path to point to the correct
    #  location.
    #  Further, it will validate that specific custom metadata is present.
    #  If this is not given, we will set the reun to debug and warn about it (this way results are still saved).
    return challenge_save_run(
        challenge=challenge,
        entry_name=entry_name,
        custom_metadata=custom_metadata,
        path=path,
        stored_filenames_relative_to=stored_filenames_relative_to,
        use_git=use_git,
        git_dirty_ignore=("results",),
        debug_folder_prefix="_",
    )
