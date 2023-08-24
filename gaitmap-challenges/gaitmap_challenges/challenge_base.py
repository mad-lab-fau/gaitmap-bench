"""Base classes and helpers to create new challenges."""

import contextlib
import copy
import json
import time
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
from tpcp import BaseTpcpObject, Dataset
from tpcp.optimize import BaseOptimize

from gaitmap_challenges._utils import _ensure_label_tuple

__all__ = [
    "BaseChallenge",
    "CvMetadata",
    "collect_cv_results",
    "collect_cv_metadata",
    "collect_opti_results",
    "save_cv_results",
    "save_opti_results",
    "load_cv_results",
    "load_opti_results",
    "resolve_dataset",
]


@dataclass(repr=False)
class BaseChallenge(BaseTpcpObject):
    """Base class for all challenges.

    Note, that we decided to use datas-classes for challenges to reduce the amount of code that needs to be written.
    This means, that you need to wrap your challenge into a dataclass decorator as well.

    Besides providing an expected interface, this class provides a `_measure_time` context manager that can be used
    within your implementation of the `run` method to measure the execution time of an algorithm.
    This will automatically set a number of results attributes about the start/end and runtime of the execution.

    Attributes
    ----------
    run_start_datetime_utc_timestamp_
        The start time of the execution of the challenge in UTC timestamp format.
    run_start_datetime_
        The start time of the execution of the challenge in ISO format.
    end_start_datetime_utc_timestamp_
        The end time of the execution of the challenge in UTC timestamp format.
    end_start_datetime_
        The end time of the execution of the challenge in ISO format.
    runtime_
        The runtime of the execution of the challenge in seconds.
    dataset_
        The instance of the dataset class actually used in the challenge.
        This is usually a copy of the dataset class instance supplied to the challenge and has potentially some
        parameters overwritten to ensure that the challenge is executed as expected.

    Other Parameters
    ----------------
    VERSION
        (Class Constant) The version of the challenge. In case of breaking changes, this should be increased.
    optimizer
        The optimizer passed to the `run` method.

    """

    run_start_datetime_utc_timestamp_: float = field(init=False)
    run_start_datetime_: str = field(init=False)
    end_start_datetime_utc_timestamp_: float = field(init=False)
    end_start_datetime_: str = field(init=False)
    runtime_: float = field(init=False)
    dataset_: Dataset = field(init=False)

    optimizer: BaseOptimize = field(init=False)

    VERSION: ClassVar[str]

    @property
    def _measure_time(self) -> Callable[[], Generator[None, None, None]]:
        """Context manager to measure the execution time of an algorithm.

        Use this within your implementation of the `run` method to measure the execution time of an algorithm.
        """

        @contextlib.contextmanager
        def timer() -> Generator[None, None, None]:
            self.run_start_datetime_utc_timestamp_ = datetime.utcnow().timestamp()
            self.run_start_datetime_ = datetime.now().astimezone().isoformat()
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            self.end_start_datetime_utc_timestamp_ = datetime.utcnow().timestamp()
            self.end_start_datetime_ = datetime.now().astimezone().isoformat()
            self.runtime_ = end_time - start_time

        return timer

    def run(self, optimizer: BaseOptimize):
        """Run the challenge."""
        raise NotImplementedError()

    def get_core_results(self) -> Dict[str, Any]:
        """Get the main results of the challenge."""
        raise NotImplementedError()

    def save_core_results(self, folder_path: Path):
        """Save the main results of the challenge to a folder."""
        raise NotImplementedError()

    @classmethod
    def load_core_results(cls, folder_path: Path) -> Dict[str, Any]:
        """Load the core results from a folder that have been stored using `save_core_results`.

        The assumption is, that the output is identical to calling `get_core_results` on the challenge instance
        directly.

        When implementing this method, make sure that it remains compatible with results saved using older versions
        of the challenge_class.
        """
        raise NotImplementedError()


def collect_cv_results(cv_results: Dict) -> pd.DataFrame:
    """Collect the cross-validation results into a pandas DataFrame.

    Use this method as part of your `get_core_results` method to simplify its implementation.
    """
    cv_results = copy.copy(cv_results)

    # This can not be properly serialized
    cv_results.pop("optimizer")

    return pd.DataFrame(cv_results)


class CvMetadata(TypedDict):
    """Metadata about the cross-validation.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset class
    dataset_columns : List[str]
        The name of the index columns of the dataset
    """

    dataset_name: str
    dataset_columns: List[str]


def collect_cv_metadata(dataset) -> CvMetadata:
    """Collect the cross-validation metadata.

    At the moment, this only concerns extracting the dataset metadata, to later reconstruct better display names for
    data labels.

    Use this method as part of your `get_core_results` method to simplify its implementation.
    """
    return {
        "dataset_name": dataset.__class__.__name__,
        "dataset_columns": list(dataset.index.columns),
    }


def collect_opti_results(cv_results: Dict) -> Optional[List[Dict[str, Any]]]:
    """Collect the optimization results.

    This will check if an optimizer object is present in the cv_results and if so, it trys to extract the best
    parameters and score from it.

    If you need more optimization information, you need to implement your own version of this.

    Use this method as part of your `get_core_results` method to simplify its implementation.
    """
    optimizer = cv_results.get("optimizer", None)
    if optimizer is None:
        return None

    opti_results = []
    for opti in optimizer:
        opti_result = {}
        if best_para := getattr(opti, "best_params_", None):
            opti_result["best_params"] = best_para
        if best_score := getattr(opti, "best_score_", None):
            opti_result["best_score"] = best_score
        opti_results.append(opti_result)

    if all(bool(o) is False for o in opti_results):
        return None

    return opti_results


def save_cv_results(
    cv_results: pd.DataFrame,
    cv_metadata: CvMetadata,
    folder_path: Union[str, Path],
    filename: str = "cv_results.json",
    meta_data_filename: str = "cv_metadata.json",
):
    """Save CV results extracted using `collect_cv_results` and `collect_cv_metadata` to a folder.

    Use this method as part of your `save_core_results` method to simplify its implementation.
    """
    cv_results.to_json(Path(folder_path) / filename)
    with (Path(folder_path) / meta_data_filename).open("w", encoding="utf8") as f:
        json.dump(cv_metadata, f, indent=4)


def save_opti_results(
    opti_results: List[Dict[str, Any]],
    folder_path: Union[str, Path],
    filename: str = "opti_results.json",
):
    """Save optimization results extracted using `collect_opti_results` to a folder.

    Use this method as part of your `save_core_results` method to simplify its implementation.
    """
    with (Path(folder_path) / filename).open("w", encoding="utf8") as f:
        json.dump(opti_results, f, cls=_NpEncoder)


def _get_dataset_tuple_class_from_metadata(metadata: Dict[str, Any]) -> Type[Tuple]:
    return namedtuple(f"{metadata['dataset_name']}DataPointLabel", metadata["dataset_columns"])


def load_cv_results(
    folder_path: Union[str, Path],
    filename: str = "cv_results.json",
    meta_data_filename: str = "cv_metadata.json",
) -> Tuple[pd.DataFrame, CvMetadata]:
    """Load the results of a cross-validation from a folder.

    This expects that the results have been saved using `save_cv_results`.

    Use this method as part of your `load_core_results` method to simplify its implementation.
    """
    with (Path(folder_path) / meta_data_filename).open(encoding="utf8") as f:
        metadata = json.load(f)

    dataset_tuple_type = _get_dataset_tuple_class_from_metadata(metadata)
    cv_results = pd.read_json(Path(folder_path) / filename)
    # We convert all columns that contain dataset labels to tuples of the correct type
    # This preserves the names of the columns
    for col in ["train_data_labels", "test_data_labels"]:
        if col in cv_results.columns:
            cv_results[col] = cv_results[col].apply(lambda x: [dataset_tuple_type(*_ensure_label_tuple(i)) for i in x])

    return cv_results, metadata


def load_opti_results(
    folder_path: Union[str, Path], filename: str = "opti_results.json"
) -> Optional[List[Dict[str, Any]]]:
    """Load the results of an optimization from a folder.

    This expects that the results have been saved using `save_opti_results`.

    Use this method as part of your `load_core_results` method to simplify its implementation.
    """
    if (path := Path(folder_path) / filename).is_file() is False:
        return None
    with path.open(encoding="utf8") as f:
        return json.load(f)


class _NpEncoder(json.JSONEncoder):
    """Json encoder that can handle numpy datatypes."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def resolve_dataset(dataset, path_or_dataset_class):
    """Resolve a dataset-instance given a path or an instance of the dataset class."""
    if isinstance(dataset, (str, Path)):
        return path_or_dataset_class(data_folder=Path(dataset))
    if isinstance(dataset, path_or_dataset_class):
        return dataset
    raise ValueError(
        f"`dataset` must either be a valid path or a valid instance of `{path_or_dataset_class.__name__}`."
    )
