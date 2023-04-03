from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score
from gaitmap_datasets import EgaitSegmentationValidation2014
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import cross_validate

from gaitmap_challenges.challenge_base import (
    BaseChallenge,
    collect_cv_results,
    collect_opti_results,
    load_cv_results,
    load_opti_results,
    save_cv_results,
    save_opti_results,
)
from gaitmap_challenges.stride_segmentation._utils import SingleValuePrecisionRecallF1

SensorNames = Literal["left_sensor", "right_sensor"]


def _final_scorer(pipeline: Pipeline, datapoint: EgaitSegmentationValidation2014, tolerance_s: float = 0.03):
    results = pipeline.safe_run(datapoint)

    matched_stride_list = evaluate_segmented_stride_list(
        ground_truth=datapoint.segmented_stride_list_,
        segmented_stride_list=results.stride_list_,
        tolerance=int(tolerance_s * datapoint.sampling_rate_hz),
    )

    combined_matched_stride_list = pd.concat(matched_stride_list)
    return {
        **precision_recall_f1_score(combined_matched_stride_list),
        "per_sample": SingleValuePrecisionRecallF1(combined_matched_stride_list),
    }


ChallengeDataset = EgaitSegmentationValidation2014


class ResultType(TypedDict):
    cv_results: pd.DataFrame
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = StratifiedKFold(n_splits=5)
    cv_params: Optional[Dict] = None

    # Class level config.
    match_tolerance_s: ClassVar[float] = 0.03

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            cv_params = {} if self.cv_params is None else self.cv_params
            # This is required to make the StratifiedKFold work
            mock_labels = self.dataset_.create_group_labels(["test", "cohort"])

            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=self.cv_iterator,
                scoring=self.get_scorer(),
                return_optimizer=True,
                mock_labels=mock_labels,
                propagate_mock_labels=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        if isinstance(self.dataset, (str, Path)):
            return ChallengeDataset(data_folder=Path(self.dataset))
        if isinstance(self.dataset, ChallengeDataset):
            return self.dataset
        raise ValueError(
            "`dataset` must either be a valid path or a valid instance of `EgaitSegmentationValidation2014`."
        )

    @classmethod
    def get_scorer(cls):
        return partial(_final_scorer, tolerance_s=cls.match_tolerance_s)

    @classmethod
    def get_imu_data(
        self,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.data

    @classmethod
    def get_reference_stride_list(
        self,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.segmented_stride_list_

    def get_core_results(self) -> ResultType:
        return {
            "cv_results": collect_cv_results(self.cv_results_),
            "opti_results": collect_opti_results(self.cv_results_),
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()
        save_cv_results(core_results["cv_results"], folder_path)
        if (opti_results := core_results["opti_results"]) is not None:
            save_opti_results(opti_results, folder_path)

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        return {
            "cv_results": load_cv_results(folder_path),
            "opti_results": load_opti_results(folder_path),
        }


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames"]
