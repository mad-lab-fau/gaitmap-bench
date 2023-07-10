import inspect
import json
import os
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from importlib.metadata import distributions
from itertools import chain
from os.path import relpath
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from cpuinfo import cpuinfo

from gaitmap_challenges.challenge_base import BaseChallenge
from gaitmap_challenges.config import config, is_debug_run


def _check_if_dirty(repo, ignore: Sequence[str] = ()):
    """Check if the repo is dirty.

    Parameters
    ----------
    repo
        The repo to check.
    ignore
        A sequence of paths to ignore.
        All paths are assumed to be relative to the repo root.

    Returns
    -------
    True if the repo is dirty, False otherwise.
    """
    if not repo.is_dirty(untracked_files=True):
        return False

    # changed files including untracked
    # Note, that we add both files of a diff (in case we have moved a file)
    files = set(chain(repo.untracked_files, *((i.a_path, i.b_path) for i in repo.index.diff(None))))
    for f in files:
        # For each file we need to check if it is in the `ignore` list or a subdirectory of a path in the `ignore` list.
        if any(Path(i) in Path(f).parents or Path(i) == Path(f) for i in ignore):
            continue
        return True
    return False


def save_run(  # noqa: PLR0912, PLR0915, C901
    challenge: BaseChallenge,
    entry_name: Union[str, Tuple[str, ...]],
    *,
    custom_metadata: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
    debug_run: bool = None,
    stored_filenames_relative_to: Optional[Union[str, Path]] = None,
    use_git: bool = True,
    git_dirty_ignore: Sequence[str] = (),
    debug_folder_prefix: str = "_",
    _force_local_debug_run_setting: bool = False,
    _caller_file_path_stack_offset: int = 0,
):
    try:
        global_config = config()
    except ValueError:
        global_config = None
    if path is None:
        if global_config is None:
            raise ValueError("No path was given and no global config is available.")
        path = global_config.results_dir
        if path is None:
            raise ValueError("No path was given and no result path is set in the global config.")

    if _force_local_debug_run_setting is False and (config_debug_run := is_debug_run()) is not None:
        if debug_run is not None:
            warnings.warn(
                f"The debug_run parameter was set ({debug_run=}), but the global config also has a debug_run setting. "
                f"The config setting ({config_debug_run=}) will be used. "
                "You should not use the `debug_run` parameter of the `save_run` function if you are using global "
                "config.",
                stacklevel=2,
            )

        debug_run = config_debug_run

    if debug_run is None:
        debug_run = False

    # We use the import path of the challenge class as the name of the challenge
    challenge_name = challenge.__class__.__module__ + "." + challenge.__class__.__name__
    challenge_version = challenge.VERSION
    # We calculate a set of internal metadata based on things we can automatically determine from the run
    metadata = {
        "entry_name": entry_name,
        "challenge_name": challenge_name,
        "challenge_version": challenge_version,
        "is_debug_run": debug_run,
        # Ideally this contains some info about the run, as all parameters should be part of the repr
        "repr_challenge": repr(challenge),
        "repr_pipeline": repr(challenge.optimizer),
        "repr_dataset": repr(challenge.dataset_),
        "dataset_name": challenge.dataset_.__class__.__name__,
        "dataset_columns": list(challenge.dataset_.index.columns),
        "run_start_datetime_utc": challenge.run_start_datetime_utc_timestamp_,
        "run_start_datetime": challenge.run_start_datetime_,
        "end_start_datetime_utc": challenge.end_start_datetime_utc_timestamp_,
        "end_start_datetime": challenge.end_start_datetime_,
        "runtime": challenge.runtime_,
        "system_info": {
            "python_version": sys.version,
            "python_implementation": sys.implementation.name,
            "packages_info": {
                d.metadata["Name"]: d.version for d in sorted(distributions(), key=lambda x: x.metadata["Name"].lower())
            },
            "cpu": {
                k: v for k, v in cpuinfo.get_cpu_info().items() if k not in ["flags", "hz_advertised", "hz_actual"]
            },
        },
    }

    # Add config
    if global_config is not None:
        metadata["config"] = global_config.to_json_dict(path_relative_to=stored_filenames_relative_to)
    else:
        metadata["config"] = None

    # Get caller filename
    frame = inspect.stack()[1 + _caller_file_path_stack_offset]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    if stored_filenames_relative_to is not None:
        filename = str(Path(filename).relative_to(stored_filenames_relative_to))
    metadata["caller_file_path"] = filename

    # Get current git commit hash and check if current git repo is dirty
    if use_git:
        import git  # pylint: disable=import-outside-toplevel

        repo = git.Repo(search_parent_directories=True)
        repo_base_dir = repo.git.rev_parse("--show-toplevel")
        if stored_filenames_relative_to is not None:
            repo_base_dir = str(relpath(Path(repo_base_dir), stored_filenames_relative_to))

        repo_is_dirty = _check_if_dirty(repo, ignore=git_dirty_ignore)
        if debug_run is False and repo_is_dirty:
            warnings.warn(
                "Trying to save results of a non-debug run from a dirty git repo. "
                "This is not allowed, as the results cannot be reproduced. "
                "We will treat the results as a debug run instead. "
                "If you are absolute sure that the results are reproducible, and this warning is an error, "
                "manually modify the metadata.json file to set is_debug_run to True and remove the debug "
                "folder prefix (if used).",
                stacklevel=2,
            )
            debug_run = True
            metadata["is_debug_run"] = debug_run

        metadata["git_commit_hash"] = repo.head.object.hexsha
        metadata["git_dirty"] = repo_is_dirty
        metadata["git_base_dir"] = repo_base_dir
        metadata["git_dirty_ignore"] = [str(p) for p in git_dirty_ignore]

    # In the target folder we create the folder structure
    # challenge_name/challenge_version/entry_name/(start_datetime _ unique_id)
    start_datetime_formatted = datetime.fromtimestamp(challenge.run_start_datetime_utc_timestamp_).strftime(
        "%Y%m%d_%H%M%S"
    )
    path = Path(path)
    if isinstance(entry_name, tuple):
        # We create nested folders
        entry_name = os.sep.join(entry_name)

    lowest_folder_name = debug_folder_prefix + start_datetime_formatted if debug_run else start_datetime_formatted

    path = path / challenge_name / challenge_version / entry_name / lowest_folder_name
    # If the file already exists, we add a number to the end
    i = 0
    while True:
        path = path.parent / f"{path.name}_{i}"
        if not path.exists():
            break
        i += 1
    path.mkdir(parents=True, exist_ok=True)

    # Save the metadata
    with (path / "custom_metadata.json").open("w", encoding="utf8") as f:
        json.dump(custom_metadata, f)
    with (path / "metadata.json").open("w", encoding="utf8") as f:
        json.dump(metadata, f)

    # Save the results
    result_path = path / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    challenge.save_core_results(result_path)

    debug_str = " (debug)" if debug_run else ""
    print(f"Saved run{debug_str} to {path.resolve()}")
    return path


ResultReturn = namedtuple("data_return", ["metadata", "results", "custom_metadata"])
