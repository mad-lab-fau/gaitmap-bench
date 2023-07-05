import copy
import json
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap_datasets import EgaitAdidas2014
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import NoAgg, cross_validate

from gaitmap_challenges.challenge_base import BaseChallenge, NpEncoder
from gaitmap_challenges.spatial_parameters._utils import SingleValueErrors

ChallengeDataset = EgaitAdidas2014


def _final_scorer(
    pipeline: Pipeline,
    datapoint: ChallengeDataset,
):
    results = pipeline.safe_run(datapoint)
    predicted = {k: v[["stride_length"]] for k, v in results.parameters_.items()}
    reference = {k: v[["stride_length"]] for k, v in datapoint.mocap_parameters_.items()}

    errors = calculate_parameter_errors(
        reference_parameter=reference,
        predicted_parameter=predicted,
        calculate_per_sensor=False,
        scoring_errors="ignore",
    )["stride_length"]

    return {
        **errors.to_dict(),
        "per_stride": SingleValueErrors(("stride_length", {"reference": reference, "predicted": predicted})),
        "stride_length_predicted": NoAgg(
            pd.concat({k: set_correct_index(v, ["s_id"]) for k, v in predicted.items()}, axis=0)["stride_length"]
        ),
        "stride_length_reference": NoAgg(
            pd.concat({k: set_correct_index(v, ["s_id"]) for k, v in reference.items()}, axis=0)["stride_length"]
        ),
    }


class ResultType(TypedDict):
    cv_results: pd.DataFrame
    stride_length: pd.DataFrame
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = field(
        default_factory=lambda: GroupKFold(n_splits=3)
    )
    cv_params: Optional[Dict] = None

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            groups = self.dataset_.create_group_labels(["participant"])
            cv_params = {} if self.cv_params is None else self.cv_params
            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=self.cv_iterator,
                groups=groups,
                scoring=self.final_scorer,
                return_optimizer=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        if isinstance(self.dataset, (str, Path)):
            return ChallengeDataset(data_folder=Path(self.dataset))
        if isinstance(self.dataset, ChallengeDataset):
            return self.dataset
        raise ValueError(f"`dataset` must either be a valid path or a valid instance of `{ChallengeDataset.__name__}`.")

    @property
    def final_scorer(self):
        return _final_scorer

    def get_imu_data(self, datapoint: ChallengeDataset) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.data

    def get_reference_segmented_stride_list(
        self,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.segmented_stride_list_

    def get_reference_parameter(
        self, datapoint: ChallengeDataset
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.mocap_parameters_

    def get_core_results(self) -> ResultType:
        cv_results = copy.copy(self.cv_results_)
        data_labels = list(chain(*cv_results["test_data_labels"]))

        raw_predictions = pd.concat(
            chain(*cv_results.pop("test_single_stride_length_predicted")), axis=0, keys=data_labels
        )
        raw_references = pd.concat(
            chain(*cv_results.pop("test_single_stride_length_reference")), axis=0, keys=data_labels
        )
        stride_length = pd.concat({"predicted": raw_predictions, "reference": raw_references}, axis=1)

        # This can not be properly serialized
        optimizer = cv_results.pop("optimizer")

        cv_results = pd.DataFrame(cv_results)

        opti_results = []
        for opti in optimizer:
            opti_result = {}
            if best_para := getattr(opti, "best_params_", None):
                opti_result["best_params"] = best_para
            if best_score := getattr(opti, "best_score_", None):
                opti_result["best_score"] = best_score
            opti_results.append(opti_result)

        if all(bool(o) is False for o in opti_results):
            opti_results = None

        return {
            "cv_results": cv_results,
            "stride_length": stride_length,
            "opti_results": opti_results,
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()

        core_results["cv_results"].to_json(folder_path / "cv_results.json")
        core_results["stride_length"].to_csv(folder_path / "stride_length.csv")
        if core_results["opti_results"] is not None:
            with (folder_path / "opti_results.json").open("w") as f:
                json.dump(core_results["opti_results"], f, cls=NpEncoder)

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        if (folder_path / "opti_results.json").is_file():
            with folder_path / "opti_results.json" as f:
                opti_results = json.load(f)
        else:
            opti_results = None
        return {
            "cv_results": pd.read_json(folder_path / "cv_results.json"),
            "stride_length": pd.read_csv(folder_path / "stride_length.csv", index_col=[0, 1]),
            "opti_results": opti_results,
        }


__all__ = ["Challenge", "ChallengeDataset", "ResultType"]
