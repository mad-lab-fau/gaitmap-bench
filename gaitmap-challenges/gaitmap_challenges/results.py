"""Methods to load and filter results from the challenge."""
import inspect
import json
import os
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
from importlib.metadata import distributions
from itertools import chain
from os.path import relpath
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import pandas as pd
from cpuinfo import cpuinfo

from gaitmap_challenges.challenge_base import BaseChallenge
from gaitmap_challenges.config import config, is_debug_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import Challenge

__all__ = [
    "save_run",
    "get_all_result_paths",
    "filter_results",
    "get_latest_result",
    "load_run",
    "load_run_metadata",
    "load_run_custom_metadata",
    "rename_keys",
    "ResultReturn",
]


def get_all_result_paths(
    challenge_class_or_full_name: Union[Type[Challenge], Challenge, str],
    base_path: Union[str, Path],
) -> Dict[Tuple[str, ...], List[Path]]:
    """Return a dictionary with the folder path for all results for the specified challenge.

    Parameters
    ----------
    challenge_class_or_full_name
        The class or an instance of the class you are searching for.
        Alternatively you can also pass the full name of the results folder of the class.
        This is usually, constructed by the module name and the class name
        (`challenge_class.__module__ + "." + challenge_class.__name__`).
        This might be helpful, if you want to load results of a challenge that has been renamed or removed.
    base_path
        The base path to search for the results.
        The folder structure will be searched recursively.

    """
    if isinstance(challenge_class_or_full_name, str):
        folder_name = challenge_class_or_full_name
    else:
        challenge_class = (
            challenge_class_or_full_name
            if isinstance(challenge_class_or_full_name, type)
            else challenge_class_or_full_name.__class__
        )
        folder_name = challenge_class.__module__ + "." + challenge_class.__name__
    try:
        folder = next(Path(base_path).rglob(folder_name))
    except StopIteration as e:
        raise ValueError(
            "No result folder for the specified challenge found found in the base folder. "
            "Did you specify the correct base-folder?\n"
            f"Currently searching in: {base_path.resolve()}\n"
            f"Based on the selected challenge the folder name we are looking for is: {folder_name}"
        ) from e

    entries = {}
    versions = {}
    for run in folder.rglob("metadata.json"):
        meta = load_run_metadata(run.parent)
        entries.setdefault(tuple(meta["entry_name"]), []).append(run.parent)
        versions.setdefault(tuple(meta["entry_name"]), []).append(meta["challenge_version"])

    if len(entries) == 0:
        raise ValueError(
            "We found the challenge folder, but no results. "
            "We are searching in the following folder: " + str(folder.resolve())
        )

    sorted_entries = {}
    for name, entry_list in entries.items():
        versions_per_entry = versions[name]
        parents_per_version = {}
        for version, entry in zip(versions_per_entry, entry_list):
            parents_per_version.setdefault(version, []).append(entry.parent)

        for version, parents in parents_per_version.items():
            if len(set(parents)) > 1:
                warnings.warn(
                    f"We found results from the same entry name ({name}) for the same challenge version "
                    f"({version}) in different folders. "
                    "This could indicate that you forgot to correctly name one of your entries when you saved a "
                    "run, or that files/folders where manually copied around. "
                    "Both can lead to issues. "
                    "Please double check your results folder.",
                    stacklevel=1,
                )
        sorted_entries[name] = sorted(entry_list, key=lambda e: e.name, reverse=False)

    return entries


def filter_results(
    results: Dict[Tuple[str, ...], List[Path]],
    challenge_version: Optional[str] = None,
    is_debug_run: Optional[bool] = None,
) -> Dict[Tuple[str, ...], List[Path]]:
    """Filter the results to only include the specified challenge version.

    Parameters
    ----------
    results
        The results to filter.
    challenge_version
        The version of the challenge to search for.
        If not specified, all results independent of the version are returned.
        If you want to search only for the current version, pass `challenge_version=Challenge.VERSION`, where
        `Challenge` is the challenge-class you are searching results for.
    is_debug_run
        If False, only real runs are returned.
        If True, only debug runs are returned.
        If None, all runs are returned.

    """
    filtered_results = {}
    for name, entry_list in results.items():
        for entry in entry_list:
            meta_data = load_run_metadata(entry)
            if challenge_version and meta_data["challenge_version"] != challenge_version:
                continue
            if is_debug_run is not None and meta_data["is_debug_run"] != is_debug_run:
                continue

            filtered_results.setdefault(name, []).append(entry)

    return filtered_results


def get_latest_result(results: Dict[Tuple[str, ...], List[Path]]) -> Dict[Tuple[str, ...], Path]:
    """Return the latest result for each entry.

    Parameters
    ----------
    results
        The dictionary of all results of all entries.

    Returns
    -------
    latest_results
        The result with the latest utc timestamp for each entry.

    """
    latest_results = {}
    for name, entry_list in results.items():
        # We sort the entries again by the actual value in the metadata.json file and not just the folder name as
        # before.
        timestamps = [load_run_metadata(e)["run_start_datetime_utc"] for e in entry_list]
        latest_results[name] = entry_list[timestamps.index(max(timestamps))]

    return latest_results


def get_metadata_as_df(
    results: Dict[Tuple[str, ...], Path],
    include_additional: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return the metadata of the results as a pandas DataFrame.

    Parameters
    ----------
    results
        The results to filter.
    include_additional
        Additional metadata to include in the DataFrame.
        This can either be a list of keys of the normal metadata or the custom metadata objects.

    """
    data = []
    for name, entry in results.items():
        meta_data = load_run_metadata(entry)
        custom_meta_data = load_run_custom_metadata(entry)
        all_meta_data = {**meta_data, **custom_meta_data}
        tmp_meta_data = {
            "entry_name": name,
            "run_start_datetime_utc": all_meta_data["run_start_datetime_utc"],
            "run_start_datetime": all_meta_data["run_start_datetime"],
            "challenge_name": all_meta_data["challenge_name"],
            "challenge_version": all_meta_data["challenge_version"],
            "is_debug_run": all_meta_data["is_debug_run"],
            **custom_meta_data,
            "path": entry,
        }

        if include_additional:
            for key in include_additional:
                tmp_meta_data[key] = all_meta_data[key]

        data.append(tmp_meta_data)
    if len(data) == 0:
        raise ValueError("No results found.")
    return pd.DataFrame(data).infer_objects()


def generate_overview_table(results: Dict[Tuple[str, ...], Path]):
    """Generate an overview table of the results.

    Parameters
    ----------
    results
        The results to display.

    Returns
    -------
    table
        A pandas DataFrame containing the overview table.

    """
    df = (
        get_metadata_as_df(results)
        .sort_values(by=["entry_name", "run_start_datetime_utc"], ascending=[True, False])
        .drop(
            columns=[
                "path",
                "is_debug_run",
                "challenge_name",
                "challenge_version",
                "run_start_datetime_utc",
                "long_description",
            ],
            errors="ignore",
        )
        .rename(
            columns={
                "entry_name": "Entry",
                "run_start_datetime": "Datetime",
                "short_description": "Description",
                "code_authors": "Code Authors",
                "algorithm_authors": "Algorithm Authors",
                "implementation_url": "Implementation",
                "references": "References",
            }
        )
        .reset_index(drop=True)
    )
    return df


@lru_cache(maxsize=100)
def load_run_metadata(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the metadata of a run.

    Parameters
    ----------
    folder_path
        The path to the folder where the run results are stored.
        The metadata is loaded from the file `metadata.json` in this folder.

    Returns
    -------
    The metadata as a dictionary.
    """
    folder_path = Path(folder_path)

    with (folder_path / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)

    return metadata


@lru_cache(maxsize=100)
def load_run_custom_metadata(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the custom metadata of a run.

    Parameters
    ----------
    folder_path
        The path to the folder where the run is stored.
        The metadata is loaded from the file `custom_metadata.json` in this folder.

    Returns
    -------
    The custom metadata as a dictionary.
    """
    folder_path = Path(folder_path)

    with (folder_path / "custom_metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)

    return metadata


ResultReturn = namedtuple("data_return", ["metadata", "results", "custom_metadata"])


def load_run(
    challenge_class: Union[Type[BaseChallenge], BaseChallenge],
    folder_path: Union[str, Path],
) -> ResultReturn:
    """Load the results of a run.

    This uses the custom load functions of a challenge to load the results.

    Parameters
    ----------
    challenge_class
        The challenge class to load the results for.
        Note, that we don't explicitly check if the challenge class matches the results.
    folder_path
        The path to the folder where the run results are stored.

    Returns
    -------
    ResultReturn
        A named tuple containing the `metadata`, the `results` and the `custom_metadata`.
        The metadata is expected to have a similar structure for all challenges.
        However, the results and the custom metadata can be arbitrary and are controlled by the challenge and the
        actual run, respectively.

    """
    challenge_class: Type[BaseChallenge] = cast(
        Type[BaseChallenge],
        challenge_class if isinstance(challenge_class, type) else challenge_class.__class__,
    )
    folder_path = Path(folder_path)

    metadata = load_run_metadata(folder_path)

    assert metadata["challenge_name"] == challenge_class.__module__ + "." + challenge_class.__name__

    results = challenge_class.load_core_results(folder_path / "results")

    # laad custom metadata
    with (folder_path / "custom_metadata.json").open(encoding="utf8") as f:
        custom_metadata = json.load(f)
    return ResultReturn(metadata, results, custom_metadata)


def rename_keys(
    in_dict: Dict[Hashable, Any],
    mapping_or_callable: Union[Dict[Hashable, Hashable], Callable[[Hashable], Hashable]],
    missing: Literal["ignore", "raise", "remove"] = "raise",
):
    """Rename the keys of a dictionary.

    Parameters
    ----------
    in_dict
        The dictionary to rename.
    mapping_or_callable
        Either a dictionary mapping the old keys to the new keys or a callable that takes the old key as input and
        returns the new key.
    missing
        What to do if a no new key can be found for a key in the dictionary.
        If "ignore", the old key is kept with its old name.
        If "raise", an exception is raised.
        If "remove", the key is removed from the dictionary.

        To trigger the "missing" case when using a callable, the callable must raise an exception.

    Returns
    -------
    renamed_dict
        The renamed dictionary.
    """
    if missing not in ["ignore", "raise", "remove"]:
        raise ValueError(f"Invalid value for missing: {missing}. Allowed values are 'ignore', 'raise' and 'remove'")

    if isinstance(mapping_or_callable, dict):
        mapping_or_callable = mapping_or_callable.get
    new_dict = {}
    for k, v in in_dict.items():
        try:
            new_key = mapping_or_callable(k)
        except:  # noqa: E722
            if missing == "raise":
                raise
            if missing == "ignore":
                new_dict[k] = v
        else:
            new_dict[new_key] = v
    return new_dict


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
    """Save the results of a challenge run.

    This is expected to be called with a challenge instance on which `run` was already executed as the first argument.

    This method stores the results exported by the `save_core_results` method of the challenge class and a large amount
    of metadata in a folder structure.
    The folder name is generated as follows:
    `{challenge_module}.{challenge_class}/{challenge_version}/{entry_name}/{timestamp}`.
    In case of debug runs, the final folder name is prefixed with the specified `debug_folder_prefix`.

    The metadata stroed with the results includes:

    - The challenge name
    - The challenge version
    - Whether this is a debug run (see Notes section)
    - A string representation of the challenge, the dataset, and the pipeline
    - Some metadata about the dataset
    - Start and end time of each run
    - Runtime
    - System Info including the Python version, the OS, the CPU and installed packages
    - The global config used for the run
    - The file path of the caller of this method
    - The git status of the repo (if available). This includes the current hash, and if the git repo is dirty
      (i.e. has uncommited changes).

    Parameters
    ----------
    challenge
        The instance of the challenge with the results to save.
    entry_name
        The name identifying the entry.
        This should be a unique name in the context of the specific challenge.
    custom_metadata
        Custom metadata to save with the run.
        Must be json serializable.
    path
        The path to the folder where the run results should be stored.
        If not given, the global config is used.
        See the `config` module to learn more about that.
    debug_run
        Whether the run should be considered a debug run.
        This setting will be overwritten by the global config, if specified there and `_force_local_debug_run_setting`
        is True.
        To understand, when a run is actually considered a debug run, see the Notes section.
    stored_filenames_relative_to
        The path relative to which the filenames in the results are stored.
        This will convert all paths in the results to be relative to this path.
        This is useful to avoid storing personal information from absolute paths in the results
    use_git
        Whether to store the status of the current git repo with the results (e.g. commit hash, dirty status, ...).
        Note, that this checks your current git repo, not the git repo of the challenge.
        The goal of saving additional metadata, is to make runs reproducible.
        This will only work, if all relevant code is actually tracked in your current git repo and not imported via
        relative paths from outside the repo.
    git_dirty_ignore
        A list of files and folders to ignore when checking if the repo is dirty.
    debug_folder_prefix
        When a run result is considered a debug run, this prefix is added to the folder name.
        This can be used to easily distinguish debug runs from normal runs (e.g. in a gitignore file).
    _force_local_debug_run_setting
        If True, the `debug_run` parameter provided to this function will be overwritten by the global config if
        provided.
        This is helpful to control the debug run setting via ENV variables, without modifying the code.
    _caller_file_path_stack_offset
        We store the path to the file that calls the `save_run` method.
        In case you wrap the `save_run` method, by your own method, this information becomes useless.
        In this case, you can set this parameter to the number of stack frames that are between your method and the
        `save_run` method, so basically to store the file that calls your method.

    Notes
    -----
    Debug runs are runs that are not considered final results.
    Runs are considered debug, if the user explicitly specifies them to be using the `debug_run=True` or via the global
    config.
    In addition, we will flag runs as debug (independent of the user setting) if the git repo is dirty, as runs
    produced with a dirty git repo, can not be reproduced and should not be considered proper benchmark results.
    This means, to produce actual runs, you need to set the explicit config to `debug_run=False` (or via the global
    config) and make sure that you don't have any uncommited changes in your git repo.
    In case, you want to exclude some files from the git dirty check, you can use the `git_dirty_ignore` parameter.
    This can be helpful, to exclude for example the `docs` or `results` folder, that will not influence the
    reproducibility of your results.

    """
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
                f"The debug_run parameter was set ({debug_run=}), but the global config also has a debug_run "
                "setting. "
                f"The config setting ({config_debug_run=}) will be used. "
                "You should not use the `debug_run` parameter of the `save_run` function if you are using global "
                "config.",
                stacklevel=2,
            )

        debug_run = config_debug_run

    if debug_run is None:
        debug_run = True

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
