"""A challenge for stride-by-stride comparison of spatial parameters on straight walks.

General Information
-------------------

Dataset
    The `EgaitParameterValidation2013 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains short walks of 100 geriatric
    patients with a GaitRite carpet as a reference system.
Sensor System
    Two IMU sensors (Shimmer 2, 102.4 Hz) are attached laterally to the shoes of the participants.
Reference System
    A GaitRite carpet (100 Hz) is used as a reference system.
    The reference system was manually synchronized with the IMU sensors on a stride level, by counting the strides
    in both systems.
    Further, the dataset contains manually annotated stride borders following the definition of Barth et al. [2]_.

Implementation Recommendations
------------------------------
The IMU data is cut to the region on the carper, where the reference system is valid.
This means, that the data starts mid-movement and there is no resting period within the signal.

The start and end value of each stride in the provided segmented stride list follow the stride definitions by Barth et
al. [2]_.
This means the start and end-values are defined by the minimum in the gyr_ml axis right before the toe-off.
The ground truth stride length is calculated from the marker position at the heel marker by calculating the distance
traveled by this marker in the ground plane between the heel strike right before a stride and the heel strike within
the stride.
This means for the first stride in the segmented stride list and for each stride after a break, no ground truth stride
length can be calculated.
The pipeline should also not calculate parameters for these strides.
To correctly handle these shifts in stride definition, you can use
:func:`~gaitmap.utils.stride_list_conversion.convert_segmented_stride_list` or check the stride ids in the calculated
ground truth parameters to remove strides without ground truth from your calculations.
Further check the `dataset example <datasets_example_>`_ for more guidance on this.
The final calculated parameters should match the provided stride list and should have the same stride ids.
Note, that we assume that parameters are calculated for each stride.
Missing strides are not handled by the evaluation.


References
----------
.. [1] Rampp, Alexander, Jens Barth, Samuel Schuelein, Karl-Gunter Gassmann, Jochen Klucken, and Bjorn M. Eskofier. ]
   “Inertial Sensor-Based Stride Parameter Calculation From Gait Sequences in Geriatric Patients.” IEEE Transactions on
   Biomedical Engineering 62, no. 4 (April 2015): 1089-97. https://doi.org/10.1109/TBME.2014.2368211.
.. [2] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
   Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence Dynamic
   Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419-40.
   https://doi.org/10.3390/s150306419.


.. _dataset_info: https://www.mad.tf.fau.de/research/datasets/#collapse_18
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/egait_parameter_validation_2013.html
.. _dataset_download: https://www.mad.tf.fau.de/research/datasets/#collapse_18

"""
import copy
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict, Union

import pandas as pd
from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap_datasets import EgaitParameterValidation2013
from sklearn.model_selection import BaseCrossValidator
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import NoAgg, cross_validate

from gaitmap_challenges.challenge_base import (
    BaseChallenge,
    CvMetadata,
    collect_cv_metadata,
    collect_opti_results,
    load_cv_results,
    load_opti_results,
    resolve_dataset,
    save_cv_results,
    save_opti_results,
)
from gaitmap_challenges.spatial_parameters._utils import SingleValueErrors

ChallengeDataset = EgaitParameterValidation2013
SensorNames = Literal["left_sensor", "right_sensor"]


def final_scorer(
    pipeline: Pipeline,
    datapoint: ChallengeDataset,
):
    """Score a pipeline build for the EgaitParameterValidation Spatial Parameter challenge on a single datapoint.

    It compares the stride length on a stride-by-stride bases calculating common error metrics.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `parameters_` after running.
        It should contain the calculated parameters for each stride.
    datapoint
        A datapoint of the EgaitParameterValidation2013 dataset.

    """
    results = pipeline.safe_run(datapoint)
    predicted = {k: v[["stride_length"]] for k, v in results.parameters_.items()}
    reference = {k: v[["stride_length"]] for k, v in datapoint.gaitrite_parameters_.items()}

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
    """The EgaitParameterValidation Spatial Parameter Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.EgaitParameterValidation2013` or a path to a directory containing
        the dataset.
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
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = 5
    cv_params: Optional[Dict] = None

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
        return final_scorer

    @classmethod
    def get_imu_data(cls, datapoint: ChallengeDataset) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.data

    @classmethod
    def get_ground_truth_segmented_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.segmented_stride_list_

    @classmethod
    def get_reference_parameter(
        cls, datapoint: ChallengeDataset
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        return datapoint.gaitrite_parameters_

    def get_core_results(self) -> ResultType:
        cv_results = copy.copy(self.cv_results_)
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

        cv_results = pd.DataFrame(cv_results)

        # This can not be properly serialized
        cv_results.pop("optimizer")

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


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "final_scorer"]
