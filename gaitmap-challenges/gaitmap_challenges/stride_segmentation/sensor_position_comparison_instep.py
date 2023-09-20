"""A challenge for stride segmentation on long @lab gait tests of healthy participants.

General Information
-------------------

Dataset
    The `Sensor Position Comparison 2019 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains 4x10m, 2x20m and a long walk
    (5 min) gait tests of 14 participants.
    The 4x10m and 2x20m tests are performed at 3 different speeds (slow, normal, fast).
    However, for this challenge, we consider all tests together as a single recording.
    This recording also contains potential walking and shuffling of the participants in between the tests.
    The dataset uses sensors at 6 different positions of each foot.
    However, for this challenge only the instep sensors are used.
Sensor System
    The instep sensors are NilsPod IMU sensors (204.8 Hz).
    The two sensors are synchronized with each other and the motion capture system with sub-sample accuracy.
Reference System
    The stride borders within the raw IMU data was labeled manually by an expert using the stride definition by [2]_.

Implementation Recommendations
------------------------------
For each participant, we only have a single recording that contains all tests.
This recording also contains all walking and movement between the tests.
Before each test, the participant was also jumping 3 times in place as a marker for the start of the test.
Further, some participants had to do some tests twice, because they did not follow the instructions.
All of this is included in the recording and all strides are labeled.

This means the algorithms should search for all strides (straight and turns) in the recording independent of everything.

References
----------
.. [1] Küderle, Arne, Nils Roth, Jovana Zlatanovic, Markus Zrenner, Bjoern Eskofier, and Felix Kluge. “The Placement of
   Foot-Mounted IMU Sensors Does Affect the Accuracy of Spatial Parameters during Regular Walking.” PLOS ONE 17, no. 6
   (June 9, 2022): e0269567. https://doi.org/10.1371/journal.pone.0269567.
.. [2] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
   Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence Dynamic
   Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419-40.
   https://doi.org/10.3390/s150306419.


.. _dataset_info: https://zenodo.org/record/5747173
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/sensor_position_comparison_2019.html
.. _dataset_download: https://zenodo.org/record/5747173
"""
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score
from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Segmentation
from sklearn.model_selection import BaseCrossValidator
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import cross_validate

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
from gaitmap_challenges.stride_segmentation._utils import SingleValuePrecisionRecallF1

SensorNames = Literal["left_sensor", "right_sensor"]


def _get_data_subset(
    datapoint_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], sensor="instep"
) -> Dict[SensorNames, pd.DataFrame]:
    return {
        "left_sensor": datapoint_data[f"l_{sensor}"],
        "right_sensor": datapoint_data[f"r_{sensor}"],
    }


def final_scorer(
    pipeline: Pipeline,
    datapoint: SensorPositionComparison2019Segmentation,
    tolerance_s: float = 0.03,
    sensor_pos: str = "instep",
):
    """Score a pipeline build for segmentation challenges of the SensorPositionComparison2019 dataset.

    It compares the reference stride list (either the original or the new one) with the stride list returned by the
    pipeline via the `stride_list_` attribute.

    It calculates the precision, recall and f1 score for the stride segmentation.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        The pipeline needs to have a `stride_list_` attribute that contains the segmented stride list after running.
    datapoint
        The datapoint of the EgaitSegmentationValidation2014 dataset.
    tolerance_s
        The tolerance in seconds that is allowed between the calculated stride borders and the reference stride
        borders.
        Both, start and end labels need to be within the tolerance to be considered a match.
    sensor_pos
        The sensor position to use for the scoring.
        This will not affect the evaluation, as the dataset stores the same ground truth for all sensor positions,
        but we include the active selection of the sensor position for completeness.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list

    """
    results = pipeline.safe_run(datapoint)

    assert hasattr(results, "stride_list_"), "The pipeline must provide its results as a `stride_list_` attribute."

    matched_stride_list = evaluate_segmented_stride_list(
        ground_truth=_get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=sensor_pos),
        segmented_stride_list=results.stride_list_,
        tolerance=int(tolerance_s * datapoint.sampling_rate_hz),
    )

    combined_matched_stride_list = pd.concat(matched_stride_list)
    return {
        **precision_recall_f1_score(combined_matched_stride_list),
        "per_sample": SingleValuePrecisionRecallF1(combined_matched_stride_list),
    }


ChallengeDataset = SensorPositionComparison2019Segmentation


class ResultType(TypedDict):
    cv_results: pd.DataFrame
    cv_metadata: CvMetadata
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    """The SensorPositionComparison Stride Segmentaion Validation Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.SensorPositionComparison2019Segmentation` or a path to a directory containing
        the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

    Other Parameters
    ----------------
    match_tolerance_s
        (Class Constant) The tolerance in seconds that is allowed between the calculated stride borders and the
        reference stride borders.
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
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = 5
    cv_params: Optional[Dict] = None

    # Class level config.
    match_tolerance_s: ClassVar[float] = 0.03
    sensor_pos: ClassVar[str] = "instep"

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: Dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            cv_params = {} if self.cv_params is None else self.cv_params

            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=self.cv_iterator,
                scoring=self.get_scorer(),
                return_optimizer=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return partial(final_scorer, tolerance_s=cls.match_tolerance_s, sensor_pos=cls.sensor_pos)

    @classmethod
    def get_imu_data(cls, datapoint: ChallengeDataset) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return _get_data_subset(datapoint.data, sensor=cls.sensor_pos)

    @classmethod
    def get_reference_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return _get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=cls.sensor_pos)

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


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames", "final_scorer"]
