"""A challenge to test the performance of a full pipeline on various laboratory tests.

Comparisons are performed by comparing the mean spatial and temporal gait parameters of the entire gait test.
Reference gait parameters are provided using a marker based motion capture system in combination with hand labeled
stride borders.
The entire validation is run as a 5-fold cross-validation to allow the algorithms to optimize parameters on an
independent train set.

General Information
-------------------

Dataset
    The `Sensor Position Comparison 2019 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains 4x10m, 2x20m and a long walk
    (5 min) gait tests of 14 participants.
    The 4x10m and 2x20m tests are performed at 3 different speeds (slow, normal, fast).
    The dataset uses sensors at 6 different positions of each foot.
    However, for this challenge only the instep sensors are used.
Sensor System
    The instep sensors are NilsPod IMU sensors (204.8 Hz).
    The two sensors are synchronized with each other and the motion capture system with sub-sample accuracy.
Reference System
    A marker-based motion capture system by Qualisys (Opus 700+ Qualisys, 28 cameras, 20x30 m capture volume) at 100 Hz
    is used to track the foot position using 4 markers per foot (at the calcaneus (CAL), at the tip of the shoe (TOE),
    and on top of the first and the fifth metatarsal (MET1 and MET5).
    For this challenge the marker at the calcaneus is used to calculate stride length.

Implementation Recommendations
------------------------------

A pipeline should only use the raw data of each datapoint and not further annotations provided by the dataset.
The strides of the Mocap system include ALL strides (including turns).
This means these strides should also be included in the algorithms output to get comparable results.
At the beginning and end of the gait test, the participants was supposed to be standing still.
However, some participants move slightly during this period.
Hence, it is recommended to validate the static period at the beginning, if this required for the algorithm and not
just assume it is there.

Notes
-----
The way the results are presented at the moment, each gait test is considered one datapoint.
However, due to the vastly different length of the tests, this might not lead to a full fair comparison.
We recommend digging deeper into the results, in case the average results for an algorithm are not as expected.

References
----------
.. [1] Küderle, Arne, Nils Roth, Jovana Zlatanovic, Markus Zrenner, Bjoern Eskofier, and Felix Kluge. “The Placement of
   Foot-Mounted IMU Sensors Does Affect the Accuracy of Spatial Parameters during Regular Walking.” PLOS ONE 17, no. 6
   (June 9, 2022): e0269567. https://doi.org/10.1371/journal.pone.0269567.

.. _dataset_info: https://zenodo.org/record/5747173
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/sensor_position_comparison_2019.html
.. _dataset_download: https://zenodo.org/record/5747173

"""
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
    collect_cv_metadata,
    collect_cv_results,
    load_cv_results,
    resolve_dataset,
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


def final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    """Score a pipeline build for the SensorPositionComparison Full Pipeline challenge on a single datapoint.

    It compares the mean gait parameters of the entire gait test between the pipeline and the reference.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `aggregated_gait_parameters_` after running.
        It should contain the aggregated gait parameters for the entire gait test.
    datapoint
        A datapoint of the SensorPositionComparison2019Mocap dataset.

    """
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
    """The SensorPositionComparison Full-Pipeline Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.SensorPositionComparison2019Mocap` or a path to a directory containing
        the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

    Other Parameters
    ----------------
    ground_truth_marker
        (Class Constant) The marker used to calculate the ground truth stride borders.
    data_padding_s
        (Class Constant) The amount of padding in seconds to add before and after each gait tests.
        This ensures that sufficient resting data is available before and after the gait test.
    sensor_pos
        (Class Constant) The sensor position to use for the challenge.

    Attributes
    ----------
    cv_results_
        The results of the cross-validation.
        This can be passed directly to the pandas DataFrame constructor to get a dataframe with the results.

    See Also
    --------
    gaitmap_challenges.challenge_base.BaseChallenge : For common parameters and attributes of all challenges.

    """

    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = GroupKFold(n_splits=5)
    cv_params: Optional[Dict] = None

    # Class level config.
    ground_truth_marker: ClassVar[Literal["toe", "fcc", "fm1", "fm5"]] = "fcc"
    data_padding_s: ClassVar[float] = 3.0
    sensor_pos: ClassVar[str] = "instep"

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
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return final_scorer

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ):
        return _get_data_subset(datapoint.data, sensor=cls.sensor_pos)

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        events = convert_segmented_stride_list(datapoint.mocap_events_, target_stride_type="ic")
        events = {f"{k}_sensor": v for k, v in events.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        marker_positions = {
            f"{foot}_sensor": datapoint.marker_position_per_stride_[f"{foot[0]}_{cls.ground_truth_marker}"].rename(
                columns=lambda x: f"pos_{x}"
            )
            for foot in ["left", "right"]
        }

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=marker_positions,
            orientations=None,
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]], axis=1) for k in ["left_sensor", "right_sensor"]}

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


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames", "final_scorer"]
