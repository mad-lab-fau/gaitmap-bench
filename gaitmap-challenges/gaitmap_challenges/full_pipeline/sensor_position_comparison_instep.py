from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterator, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.utils.stride_list_conversion import convert_segmented_stride_list
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Mocap,
)
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import NoAgg, cross_validate

from gaitmap_challenges.challenge_base import (
    BaseChallenge,
    CvMetadata,
    _resolve_dataset,
    collect_cv_metadata,
    collect_cv_results,
    load_cv_results,
    save_cv_results,
)
from gaitmap_challenges.full_pipeline._utils import ParameterErrors

SensorNames = Literal["left_sensor", "right_sensor"]

ChallengeDataset = SensorPositionComparison2019Mocap


def _get_data_subset(
    datapoint_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], sensor="instep"
) -> Dict[SensorNames, pd.DataFrame]:
    return {
        "left_sensor": datapoint_data[f"l_{sensor}"],
        "right_sensor": datapoint_data[f"r_{sensor}"],
    }


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
    cv_metadata: CvMetadata


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = GroupKFold(n_splits=5)
    cv_params: Optional[Dict] = None

    # Class level config.
    ground_truth_marker: ClassVar[Literal["toe", "fcc", "fm1", "fm5"]] = "fcc"
    data_padding_s: ClassVar[float] = 3.0

    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            self.dataset_ = self.dataset_.clone().set_params(data_padding_s=self.data_padding_s)
            cv_params = {} if self.cv_params is None else self.cv_params

            # This is used to ensure that repetitions of the same patient are always in the same fold
            mock_groups = self.dataset_.create_group_labels(["participant"])

            self.cv_results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset_,
                groups=mock_groups,
                cv=self.cv_iterator,
                scoring=self.get_scorer(),
                return_optimizer=True,
                propagate_groups=True,
                **cv_params,
            )

    def _resolve_dataset(self):
        return _resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return _final_scorer

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ):
        return _get_data_subset(datapoint.data)

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        events = convert_segmented_stride_list(datapoint.mocap_events_, target_stride_type="ic")
        events = {f"{k}_sensor": v for k, v in events.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        trajectory = {
            foot: datapoint.marker_position_[f"{foot[0]}_{cls.ground_truth_marker}"].rename(
                columns=lambda x: f"pos_{x}"
            )
            for foot in ["left_sensor", "right_sensor"]
        }

        per_stride_trajectory = marker_position_per_stride_(
            trajectory,
            events,
        )

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=per_stride_trajectory,
            orientations=None,
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]]) for k in ["left_sensor", "right_sensor"]}

    @classmethod
    def get_aggregated_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        return pd.concat(cls.get_ground_truth_parameter(datapoint)).mean()

    def get_core_results(self) -> ResultType:
        return {
            "cv_results": collect_cv_results(self.cv_results_),
            "cv_metadata": collect_cv_metadata(self.dataset),
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()
        save_cv_results(core_results["cv_results"], core_results["cv_metadata"], folder_path)

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        cv_results = load_cv_results(folder_path)
        return {"cv_results": cv_results[0], "cv_metadata": cv_results[1]}


# TODO: remove once implemented in gaitmap datasets
def marker_position_per_stride_(trajectory, mocap_events):
    per_stride_trajectory = {}
    for foot, events in mocap_events.items():
        output_per_foot = {}
        data = trajectory[foot]
        for s_id, stride in events.iterrows():
            # This cuts out the n+1 samples for each stride.
            # The first sample is the value before the stride started.
            # This is the equivalent to the "initial" position/orientation
            output_per_foot[s_id] = data.iloc[int(stride["start"]) : int(stride["end"] + 1)].reset_index(drop=True)
        per_stride_trajectory[foot] = pd.concat(output_per_foot, names=["s_id", "sample"])
    return per_stride_trajectory
