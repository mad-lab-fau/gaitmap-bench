import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Type, Union, Dict, Tuple, List, Any, cast, Optional, Sequence

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
        parents = set(e.parent for e in entry_list)
        if len(parents) > 1:
            warnings.warn(
                f"We found results from the same entry name ({name}) in different folders. "
                "This could indicate that you forgot to correctly name one of your entries when you saved a "
                "run, or that files/folders where manually copied around. "
                "Both can lead to issues. "
                "Please double check your results folder."
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
        The results to filter.

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

    return pd.DataFrame(data).infer_objects()


def generate_overview_table(results: Dict[Tuple[str, ...], Path]):
    """Generate an overview table of the results.

    Parameters
    ----------
    results
        The results to filter.

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
            ]
        )
        .rename(
            columns={
                "entry_name": "Entry",
                "run_start_datetime": "Datetime",
                "short_description": "Description",
                "code_authors": "Code Authors",
                "algorithm_authors": "Algorithm Authors",
                "implementation_link": "Implementation",
                "references": "References",
            }
        )
        .reset_index(drop=True)
    )
    return df


@lru_cache(maxsize=100)
def load_run_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)

    with open(path / "metadata.json", encoding="utf8") as f:
        metadata = json.load(f)

    return metadata


@lru_cache(maxsize=100)
def load_run_custom_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)

    with open(path / "custom_metadata.json", encoding="utf8") as f:
        metadata = json.load(f)

    return metadata


def load_run(challenge_class: Union[Type[BaseChallenge], BaseChallenge], path: Union[str, Path]) -> ResultReturn:
    challenge_class: Type[BaseChallenge] = cast(
        Type[BaseChallenge], challenge_class if isinstance(challenge_class, type) else challenge_class.__class__
    )
    path = Path(path)

    metadata = load_run_metadata(path)

    assert metadata["challenge_name"] == challenge_class.__module__ + "." + challenge_class.__name__

    results = challenge_class.load_core_results(path / "results")

    # laad custom metadata
    with open(path / "custom_metadata.json", encoding="utf8") as f:
        custom_metadata = json.load(f)
    return ResultReturn(metadata, results, custom_metadata)
