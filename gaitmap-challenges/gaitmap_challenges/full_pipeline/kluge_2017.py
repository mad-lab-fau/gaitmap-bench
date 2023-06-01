from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Optional, Union, Iterator, Dict

import pandas as pd
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
from gaitmap.utils.consts import GF_POS
from gaitmap_datasets import Kluge2017
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from gaitmap.evaluation_utils import calculate_parameter_errors

from tpcp.validate import cross_validate, NoAgg

from gaitmap_challenges.challenge_base import BaseChallenge, collect_cv_results, save_cv_results, load_cv_results
from gaitmap_challenges.full_pipeline._utils import ParameterErrors

ChallengeDataset = Kluge2017


def _final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    results = pipeline.safe_run(datapoint)

    aggregated_paras = results.aggregated_gait_parameters_
    aggregated_ground_truth = Challenge.get_aggregated_ground_truth_parameter(datapoint)

    errors = calculate_parameter_errors(
        predicted_parameter=aggregated_paras.to_frame().T.rename_axis(index="test"),
        reference_parameter=aggregated_ground_truth.to_frame().T.rename_axis(index="test"),
        id_column="test",
    )[0].iloc[0]

    errors.index = [f"{parameter}__{metric}" for metric, parameter in errors.index]

    return {
        **{k: NoAgg(v) for k, v in errors.to_dict().items()},
        "agg": ParameterErrors({"predicted": aggregated_paras, "reference": aggregated_ground_truth}),
    }


class ResultType(TypedDict):
    cv_results: pd.DataFrame


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = StratifiedGroupKFold(n_splits=5)
    cv_params: Optional[Dict] = None

    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            cv_params = {} if self.cv_params is None else self.cv_params

            # This is used to stratify the data by patient
            mock_labels = self.dataset_.create_group_labels(["patient"])

            # This is used to ensure that repetitions of the same patient are always in the same fold
            mock_groups = self.dataset_.create_group_labels(["participant"])

            self.cv_results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset_,
                groups=mock_groups,
                cv=self.cv_iterator,
                scoring=self.get_scorer(),
                return_optimizer=True,
                mock_labels=mock_labels,
                propagate_mock_labels=True,
                propagate_groups=True,
                **cv_params,
            )

    def _resolve_dataset(self):
        if isinstance(self.dataset, (str, Path)):
            return ChallengeDataset(data_folder=Path(self.dataset))
        if isinstance(self.dataset, ChallengeDataset):
            return self.dataset
        raise ValueError(f"`dataset` must either be a valid path or a valid instance of `{ChallengeDataset.__name__}`.")

    @classmethod
    def get_scorer(cls):
        return _final_scorer

    @classmethod
    def get_imu_data(
        self,
        datapoint: ChallengeDataset,
    ):
        return datapoint.data

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        fake_events = {k: ev.assign(min_vel=pd.NA) for k, ev in datapoint.mocap_events_.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=fake_events, sampling_rate_hz=datapoint.mocap_sampling_rate_hz
        )

        per_stride_trajectory = {k: v["ankle"][GF_POS] for k, v in datapoint.marker_position_per_stride_.items()}

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=per_stride_trajectory,
            orientations=None,
            stride_event_list=fake_events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]]) for k in ["left", "right"]}

    @classmethod
    def get_aggregated_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        return pd.concat(cls.get_ground_truth_parameter(datapoint)).mean()

    def get_core_results(self) -> ResultType:
        return {
            "cv_results": collect_cv_results(self.cv_results_),
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()
        save_cv_results(core_results["cv_results"], folder_path)

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        return {
            "cv_results": load_cv_results(folder_path),
        }
