"""A challenge for stride segmentation from continuous IMU data @Lab and in simulated @home environments.

General Information
-------------------

Dataset
    The `Egait Segmentation Validation 2014 <dataset_info_>`_ dataset [1]_ is used for this challenge
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_).
    It contains 4x10m gait tests from 30 participants (10 controls, 10 Parkinson's disease, 10 geriatric patients) and
    15 simulated @home tests from 5 participants per cohort (5 controls, 5 Parkinson's disease, 5 geriatric patients ).
Sensor System
    All participants wore Shimmer2R (102.4 Hz) IMUs laterally on both shoes.
Reference System
    All trials where recorded with a video camera to make the original annotations (not used in this challenge).
    In addition, the data was expert labeled based on the raw gyro data.
    The original annotations explicitly excluded turns and stairs.
    The new annotations included all gait like movements (turns, stairs, etc.).
    This challenge uses the new annotations.
    For a challenge using the original annotations, see
    :mod:`~gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014_original_label`.
    All annotations where performed following the stride definition by [1]_.

Implementation Recommendations
------------------------------

This challenge version (using the new annotations) expects algorithms to be able to segment all types of strides
including turns and stairs.
This means algorithms likely need be more sensitive.

The data does only contain gait and no other movements.
This should make this a relatively easy dataset for gait segmentation.

References
----------
.. [1] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
       Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence
       Dynamic Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419-40.
       https://doi.org/10.3390/s150306419.

.. _dataset_info: https://www.mad.tf.fau.de/research/datasets/#collapse_18
.. _datasets_example:
    https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/egait_segmentation_validation_2014.html
.. _dataset_download: https://www.mad.tf.fau.de/research/datasets/#collapse_18

"""
from dataclasses import dataclass, field
from functools import partial
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
from gaitmap.evaluation_utils import (
    evaluate_segmented_stride_list,
    precision_recall_f1_score,
)
from gaitmap_datasets import EgaitSegmentationValidation2014
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
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


def final_scorer(
    pipeline: Pipeline,
    datapoint: EgaitSegmentationValidation2014,
    tolerance_s: float = 0.03,
    use_original_labels: bool = False,
):
    """Score a pipeline build for segmentation challenges of the Egait Segmentation Validation 2014 dataset.

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
    use_original_labels
        If True, the original stride labels are used for the reference stride list.
        The original labels only contain straight strides.
        If False, the new labels are used for the reference stride list.
        The new labels contain all strides including turns and stairs.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list

    """
    results = pipeline.safe_run(datapoint)

    reference = datapoint.segmented_stride_list_original_ if use_original_labels else datapoint.segmented_stride_list_

    matched_stride_list = evaluate_segmented_stride_list(
        ground_truth=reference,
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
    cv_metadata: CvMetadata
    opti_results: Optional[List[Dict[str, Any]]]


@dataclass(repr=False)
class Challenge(BaseChallenge):
    """The EgaitSegmentation Validation Challenge.

    This challenge uses the new labels by default.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.EgaitSegmentationValidation2014` or a path to a directory containing
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
    use_original_labels
        (Class Constant) If True, the original stride labels are used for the reference stride list.
        The original labels only contain straight strides.
        If False, the new labels are used for the reference stride list.
        The new labels contain all strides including turns and stairs.

    Attributes
    ----------
    cv_results_
        The results of the cross-validation.
        This can be passed directly to the pandas DataFrame constructor to get a dataframe with the results.

    See Also
    --------
    gaitmap_challenges.challenge_base.BaseChallenge : For common parameters and attributes of all challenges.
    gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014_original_label.Challenge : The same
        challenge, but with the original labels

    """

    dataset: Optional[Union[str, Path, ChallengeDataset]]
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = StratifiedKFold(n_splits=5)
    cv_params: Optional[Dict] = None

    # Class level config.
    match_tolerance_s: ClassVar[float] = 0.03
    use_original_labels: ClassVar[bool] = False

    # Update the version, when the challenge_class is changed in a relevant way
    # Note: Remember to update the version in the original_label class as well.
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
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        return partial(
            final_scorer,
            tolerance_s=cls.match_tolerance_s,
            use_original_labels=cls.use_original_labels,
        )

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.data

    @classmethod
    def get_reference_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        if cls.use_original_labels:
            return datapoint.segmented_stride_list_original_
        return datapoint.segmented_stride_list_

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
