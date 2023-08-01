from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

import pandas as pd
from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap.utils.stride_list_conversion import convert_segmented_stride_list
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Mocap,
)
from sklearn.model_selection import BaseCrossValidator
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
    save_cv_results,
    save_opti_results,
)
from gaitmap_challenges.spatial_parameters._utils import SingleValueErrors

ChallengeDataset = SensorPositionComparison2019Mocap
SensorNames = Literal["left_sensor", "right_sensor"]


def _get_data_subset(
    datapoint_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], sensor="instep"
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    return {
        "left_sensor": datapoint_data[f"l_{sensor}"],
        "right_sensor": datapoint_data[f"r_{sensor}"],
    }


def _final_scorer(
    pipeline: Pipeline,
    datapoint: ChallengeDataset,
):
    results = pipeline.safe_run(datapoint)
    predicted = {k: v[["stride_length"]] for k, v in results.parameters_.items()}
    # Calling the Challenge method here is not optimal, as it couples this function to the Challenge class.
    # But we keep it for now, until it creates a problem.
    reference = {k: v[["stride_length"]] for k, v in Challenge.get_ground_truth_parameter(datapoint).items()}

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
    cv_metadata: CvMetadata
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = 5
    cv_params: Optional[Dict] = None

    # Class level config.
    sensor_pos: ClassVar[str] = "instep"
    ground_truth_marker: ClassVar[Literal["toe", "fcc", "fm1", "fm5"]] = "fcc"
    data_padding_s: ClassVar[float] = 3.0

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            self.dataset_ = self.dataset_.clone().set_params(data_padding_s=self.data_padding_s)
            cv_params = {} if self.cv_params is None else self.cv_params

            # This is used to ensure that tests from the same are always in the same fold
            mock_groups = self.dataset_.create_group_labels(["participant"])

            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=self.cv_iterator,
                scoring=self.get_scorer(),
                return_optimizer=True,
                groups=mock_groups,
                propagate_groups=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        if isinstance(self.dataset, (str, Path)):
            return ChallengeDataset(data_folder=Path(self.dataset))
        if isinstance(self.dataset, ChallengeDataset):
            return self.dataset
        raise ValueError(
            "`dataset` must either be a valid path or a valid instance of `SensorPositionComparison2019Mocap`."
        )

    @classmethod
    def get_scorer(cls):
        return _final_scorer

    @classmethod
    def get_imu_data(cls, datapoint: ChallengeDataset) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return _get_data_subset(datapoint.data, sensor=cls.sensor_pos)

    @classmethod
    def get_ground_truth_segmented_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return _get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=cls.sensor_pos)

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        events = convert_segmented_stride_list(datapoint.mocap_events_, target_stride_type="ic")
        events = {f"{k}_sensor": v for k, v in events.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        marker_positions = {
            f"{foot}_sensor": datapoint.marker_position_per_stride_[foot][
                f"{foot[0]}_{cls.ground_truth_marker}"
            ].rename(columns=lambda x: f"pos_{x}")
            for foot in ["left", "right"]
        }

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=marker_positions,
            orientations=None,
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]], axis=1) for k in ["left_sensor", "right_sensor"]}

    def get_core_results(self) -> ResultType:
        return {
            "cv_results": collect_cv_results(self.cv_results_),
            "cv_metadata": collect_cv_metadata(self.dataset_),
            "opti_results": collect_opti_results(self.cv_results_),
        }

    def save_core_results(self, folder_path) -> None:
        core_results = self.get_core_results()
        save_cv_results(core_results["cv_results"], core_results["cv_metadata"], folder_path)
        if (opti_results := core_results["opti_results"]) is not None:
            save_opti_results(opti_results, folder_path)

    @classmethod
    def load_core_results(cls, folder_path) -> ResultType:
        cv_results = load_cv_results(folder_path)
        return {
            "cv_results": cv_results[0],
            "cv_metadata": cv_results[1],
            "opti_results": load_opti_results(folder_path),
        }


__all__ = ["Challenge", "ChallengeDataset", "ResultType"]
