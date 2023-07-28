"""Methods to load and filter results from the challenge."""

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional, Sequence, Tuple, Type, Union, cast

import pandas as pd

from gaitmap_challenges._base import ResultReturn
from gaitmap_challenges.challenge_base import BaseChallenge
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import Challenge


def get_all_results_path(
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
    for run in folder.rglob("metadata.json"):
        meta = load_run_metadata(run.parent)
        entries.setdefault(tuple(meta["entry_name"]), []).append(run.parent)

    if len(entries) == 0:
        raise ValueError(
            "We found the challenge folder, but no results. "
            "We are searching in the following folder: " + str(folder.resolve())
        )

    sorted_entries = {}
    for name, entry_list in entries.items():
        parents = {e.parent for e in entry_list}
        if len(parents) > 1:
            warnings.warn(
                f"We found results from the same entry name ({name}) in different folders. "
                "This could indicate that you forgot to correctly name one of your entries when you saved a "
                "run, or that files/folders where manually copied around. "
                "Both can lead to issues. "
                "Please double check your results folder.",
                stacklevel=2,
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
    results: Dict[Tuple[str, ...], Path], include_additional: Optional[Sequence[str]] = None
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


def load_run(challenge_class: Union[Type[BaseChallenge], BaseChallenge], folder_path: Union[str, Path]) -> ResultReturn:
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
        Type[BaseChallenge], challenge_class if isinstance(challenge_class, type) else challenge_class.__class__
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
