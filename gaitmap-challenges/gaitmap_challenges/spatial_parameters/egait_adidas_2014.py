from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap_datasets import EgaitAdidas2014
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import NoAgg, cross_validate

from gaitmap_challenges.challenge_base import (
    BaseChallenge,
    CvMetadata,
    collect_cv_metadata,
    collect_cv_results,
    collect_opti_results,
    load_cv_results,
    load_opti_results,
    resolve_dataset,
    save_cv_results,
    save_opti_results,
)
from gaitmap_challenges.spatial_parameters._utils import SingleValueErrors

ChallengeDataset = EgaitAdidas2014

SensorNames = Literal["left_sensor", "right_sensor"]


def _final_scorer(
    pipeline: Pipeline,
    datapoint: ChallengeDataset,
):
    results = pipeline.safe_run(datapoint)
    predicted = {k: v[["stride_length"]] for k, v in results.parameters_.items()}
    reference = {k: v[["stride_length"]] for k, v in datapoint.mocap_parameters_.items()}

    errors = calculate_aggregated_parameter_errors(
        reference_parameter=reference,
        predicted_parameter=predicted,
        calculate_per_sensor=False,
        scoring_errors="ignore",
    )["stride_length"]

    return {
        **errors.to_dict(),
        "per_stride": SingleValueErrors(("stride_length", {"reference": reference, "predicted": predicted})),
        "stride_length_predicted": NoAgg(
            pd.concat(
                {k: set_correct_index(v, ["s_id"]) for k, v in predicted.items()},
                axis=0,
            )["stride_length"]
        ),
        "stride_length_reference": NoAgg(
            pd.concat(
                {k: set_correct_index(v, ["s_id"]) for k, v in reference.items()},
                axis=0,
            )["stride_length"]
        ),
    }


class ResultType(TypedDict):
    cv_results: pd.DataFrame
    stride_length: pd.DataFrame
    cv_metadata: CvMetadata
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = GroupKFold(n_splits=5)
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
                scoring=self.get_scorer(),
                return_optimizer=True,
                propagate_groups=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return _final_scorer

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ):
        return datapoint.data

    @classmethod
    def get_ground_truth_segmented_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.segmented_stride_list_

    @classmethod
    def get_ground_truth_parameter(
        self, datapoint: ChallengeDataset
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.mocap_parameters_

    def get_core_results(self) -> ResultType:
        cv_results = collect_cv_results(self.cv_results_)
        data_labels = list(chain(*cv_results["test_data_labels"]))

        raw_predictions = pd.concat(
            chain(*cv_results.pop("test_single_stride_length_predicted")),
            axis=0,
            keys=data_labels,
        )
        raw_references = pd.concat(
            chain(*cv_results.pop("test_single_stride_length_reference")),
            axis=0,
            keys=data_labels,
        )
        stride_length = pd.concat({"predicted": raw_predictions, "reference": raw_references}, axis=1)

        return {
            "cv_results": cv_results,
            "cv_metadata": collect_cv_metadata(self.dataset_),
            "stride_length": stride_length,
            "opti_results": collect_opti_results(self.cv_results_),
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()

        save_cv_results(core_results["cv_results"], core_results["cv_metadata"], folder_path)
        if (opti_results := core_results["opti_results"]) is not None:
            save_opti_results(opti_results, folder_path)

        core_results["stride_length"].to_csv(folder_path / "stride_length.csv")

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        cv_results = load_cv_results(folder_path)
        return {
            "cv_results": cv_results[0],
            "cv_metadata": cv_results[1],
            "opti_results": load_opti_results(folder_path),
            "stride_length": pd.read_csv(folder_path / "stride_length.csv", index_col=[0, 1]),
        }


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames"]
