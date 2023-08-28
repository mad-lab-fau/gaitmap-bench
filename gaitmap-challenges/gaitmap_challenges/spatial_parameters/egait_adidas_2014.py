"""A challenge for stride-by-stride comparison of spatial parameters on simple lab tests.

General Information
-------------------

Dataset
    The `Egait Adidas 2014 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains data from 20 participants
    with motion capture reference.
    Each participant performed multiple trials with different self-selected stride lengths and speed.
Sensor System
    Depending on the trial either a Shimmer2R (102.4 Hz) or a Shimmer3 (204.8 Hz) was used.
Reference System
    A 16 camera Vicon system was used to record the motion capture reference.
    On each shoe 6 markers were placed.
    The reference system was synchronized with the sensor system using a trigger signal generated by a light-barrier
    at the beginning of the capture volume.

Implementation Recommendations
------------------------------
The IMU data is cut to the tigger signal.
This means usually, the signal starts mid-movement and there is no resting phases within the signal.
The motion capture system only covers a small part within each trial (usually 2-3 strides).
The provided segmented stride list also only covers this part.
The pipeline is only expected to return results for these strides.
The start and end value of each stride in the provided segmented stride list follow the stride definitions by Barth et al. [2]_.
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
The final calculated parameters should match the provided stride list and should have the same stride ids.
Note, that we assume that parameters are calculated for each stride.
Missing strides are not handled by the evaluation.

Depending on the trial a different sensor system is used.
This means, the sampling rate of the sensor data is different.
Algorithms need to be able to handle this.

References
----------
.. [1] Kanzler, Christoph M., Jens Barth, Alexander Rampp, Heiko Schlarb, Franz Rott, Jochen Klucken, and
   Bjoern M. Eskofier. “Inertial Sensor Based and Shoe Size Independent Gait Analysis Including Heel and Toe Clearance
   Estimation.” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
   Society, EMBS 2015-Novem (2015): 5424–27. https://doi.org/10.1109/EMBC.2015.7319618.
.. [2] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
   Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence Dynamic
   Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419–40.
   https://doi.org/10.3390/s150306419.

.. _dataset_info: https://osf.io/qjm8y/
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/egait_adidas_2014.html
.. _dataset_download: https://osf.io/qjm8y/

"""
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


def final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    """Score a pipeline build for the EgaitAdidas Spatial Parameter challenge on a single datapoint.

    It compares the stride length on a stride-by-stride bases calculating common error metrics.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `parameters_` after running.
        It should contain the calculated parameters for each stride.
    datapoint
        A datapoint of the EgaitAdidas2014 dataset.

    """
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
    """The EgaitAdidas Spatial Parameter Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.EgaitAdidas2014` or a path to a directory containing
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
    cv_iterator: Optional[Union[int, BaseCrossValidator, Iterator]] = GroupKFold(n_splits=5)
    cv_params: Optional[Dict] = None

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
        return final_scorer

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
        cls, datapoint: ChallengeDataset
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


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames", "final_scorer"]
