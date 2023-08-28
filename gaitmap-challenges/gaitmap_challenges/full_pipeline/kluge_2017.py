"""A challenge to test the performance of a full gaitanalysis pipeline on laboratory 4x10m gait tests.

Comparisons are performed on the calculated means of spatial temporal parameters over the entire gait test.
Reference gait parameters are provided by a marker-less motion capture system that covers the center of the walking
section.
The entire validation is run as a 5-fold cross-validation to allow the algorithms to optimize parameters on an
independent train set.

General Information
-------------------

Dataset
    The `Kluge 2017 <dataset_info_>`_  dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains 4x10m gait tests of 20 TODO
Sensor System
    Two IMU sensors (Shimmer 3, 102.4 Hz) are attached laterally to the shoes of the participants.
Reference System
    A marker-less motion capture system (Simi Motion, 100 Hz) is used to track the foot and ankle trajectory.
    For this challenge, we use the trajectory of the ankle marker to calculate stride length.
    Heel strikes were labeled manually based on the video data.

Implementation Recommendations
------------------------------
As the MoCap system does not cover the turns at the end of each 10m section algorithms should also cut of all turning
strides (recommendation: turning angle > 20 deg) before calculating the mean and the variance.
Otherwise, calculated results will have a considerable bias.

Notes
-----
The comparison here is fundamentally different that the validation performed in the original paper [1]_.
There, parameters were compared on a per-stride basis, while here the mean over the entire gait test is used.
Hence, error values are not directly comparable.

References
----------
.. [1] Kluge, Felix, Heiko Gaßner, Julius Hannink, Cristian Pasluosta, Jochen Klucken, and Björn M. Eskofier.
     “Towards Mobile Gait Analysis: Concurrent Validity and Test-Retest Reliability of an Inertial Measurement System
     for the Assessment of Spatio-Temporal Gait Parameters.” Sensors 17, no. 7 (July 2017): 1522.
     https://doi.org/10.3390/s17071522.

.. _dataset_info: https://www.mad.tf.fau.de/research/datasets/#collapse_13
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/kluge_2017.html
.. _dataset_download: https://osf.io/cfb7e/

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.utils.consts import GF_POS
from gaitmap_datasets import Kluge2017
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold
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

__all__ = ["Challenge", "ChallengeDataset", "final_scorer", "ResultType"]

ChallengeDataset = Kluge2017


def final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    """Score a pipeline build for the Kluge2017 challenge on a single datapoint.

    It compares the mean gait parameters of the entire gait test between the pipeline and the reference.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `aggregated_gait_parameters_` after running.
        It should contain the aggregated gait parameters for the entire gait test.
    datapoint
        A datapoint of the Kluge2017 dataset.

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
    """The Kluge2017 Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.Kluge2017` or a path to a directory containing the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

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
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return final_scorer

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ):
        return datapoint.data

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        fake_events = {k: ev.assign(min_vel=pd.NA) for k, ev in datapoint.mocap_events_.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=fake_events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        per_stride_trajectory = {k: v["ankle"][GF_POS] for k, v in datapoint.marker_position_per_stride_.items()}

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=per_stride_trajectory,
            orientations=None,
            stride_event_list=fake_events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]]) for k in ["left", "right"]}

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
