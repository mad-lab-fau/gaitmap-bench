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


@dataclass(repr=False)
class BaseChallenge(BaseTpcpObject):
    run_start_datetime_utc_timestamp_: float = field(init=False)
    run_start_datetime_: str = field(init=False)
    end_start_datetime_utc_timestamp_: float = field(init=False)
    end_start_datetime_: str = field(init=False)
    runtime_: float = field(init=False)
    dataset_: Dataset = field(init=False)

    optimizer: BaseOptimize = field(init=False)

    NAME: ClassVar[str]
    VERSION: ClassVar[str]

    @property
    def _measure_time(self) -> Callable[[], Generator[None, None, None]]:
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
        raise NotImplementedError()

    def get_core_results(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def save_core_results(self, folder_path):
        raise NotImplementedError()

    @classmethod
    def load_core_results(cls, folder_path) -> Dict[str, Any]:
        """Load the core results from a folder that have been stored using `save_core_results`.

        When implementing this method, make sure that it remains compatible with results saved using older versions
        of the challenge_class.
        """
        raise NotImplementedError()


def collect_cv_results(cv_results: Dict) -> pd.DataFrame:
    cv_results = copy.copy(cv_results)

    # This can not be properly serialized
    cv_results.pop("optimizer")

    return pd.DataFrame(cv_results)


class CvMetadata(TypedDict):
    dataset_name: str
    dataset_columns: List[str]


def collect_cv_metadata(dataset) -> CvMetadata:
    return {
        "dataset_name": dataset.__class__.__name__,
        "dataset_columns": list(dataset.index.columns),
    }


def collect_opti_results(cv_results: Dict) -> Optional[List[Dict[str, Any]]]:
    cv_results = copy.copy(cv_results)

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
    cv_results.to_json(Path(folder_path) / filename)
    with (Path(folder_path) / meta_data_filename).open("w", encoding="utf8") as f:
        json.dump(cv_metadata, f, indent=4)


def save_opti_results(
    opti_results: List[Dict[str, Any]],
    folder_path: Union[str, Path],
    filename: str = "opti_results.json",
):
    with (Path(folder_path) / filename).open("w", encoding="utf8") as f:
        json.dump(opti_results, f, cls=NpEncoder)


def _get_dataset_tuple_class_from_metadata(metadata: Dict[str, Any]) -> Type[Tuple]:
    return namedtuple(metadata["dataset_name"], metadata["dataset_columns"])


def load_cv_results(
    folder_path: Union[str, Path],
    filename: str = "cv_results.json",
    meta_data_filename: str = "cv_metadata.json",
) -> Tuple[pd.DataFrame, CvMetadata]:
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
    if (path := Path(folder_path) / filename).is_file() is False:
        return None
    with path.open(encoding="utf8") as f:
        return json.load(f)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _resolve_dataset(dataset, dataset_class):
    if isinstance(dataset, (str, Path)):
        return dataset_class(data_folder=Path(dataset))
    if isinstance(dataset, dataset_class):
        return dataset
    raise ValueError(f"`dataset` must either be a valid path or a valid instance of `{dataset_class.__name__}`.")
