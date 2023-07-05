import warnings
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from gaitmap.base import BaseTrajectoryMethod
from gaitmap.event_detection import RamppEventDetection
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.trajectory_reconstruction import (
    RegionLevelTrajectory,
    RtsKalman,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from sklearn.model_selection import ParameterGrid
from tpcp import Pipeline
from tpcp.optimize import GridSearch
from typing_extensions import Self

# from gaitmap_challenges import Config
#
# Config()

SensorNames = Literal["left_sensor", "right_sensor"]


dataset = ChallengeDataset(
    data_folder=Path(
        "/home/arne/Documents/repos/work/datasets/ValidationDatasets-GaitAnalysis/MoCapReference_Adidas/public_version/"
    ),
    memory=Memory("../.cache"),
).get_subset(sensor="shimmer3")
print(len(dataset))

challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})


class Entry(Pipeline[ChallengeDataset]):
    # Result object
    parameters_: Dict[SensorNames, pd.DataFrame]
    event_list_: Dict[SensorNames, pd.DataFrame]
    position_: Dict[SensorNames, pd.DataFrame]
    orientation_: Dict[SensorNames, pd.DataFrame]

    def __init__(
        self,
        traj_method: BaseTrajectoryMethod,
        search_region_padding_s: float = 3.0,
    ):
        self.traj_method = traj_method
        self.search_region_padding_s = search_region_padding_s

    def run(self, datapoint: ChallengeDataset) -> Self:
        data = challenge.get_imu_data(datapoint)
        foot_sensors = {n.split("_")[0] + "_like": n for n in get_multi_sensor_names(data)}
        data_bf = convert_to_fbf(data, **foot_sensors)
        stride_list = challenge.get_reference_segmented_stride_list(datapoint)
        sampling_rate_hz = datapoint.sampling_rate_hz
        # 1. We only have segmented strides -> we need to find the min_vel for each stride. For this we use Rampp
        # Event Detection, but only calculate the min_vel.
        min_vel_stride_list = (
            RamppEventDetection(detect_only=("min_vel",))
            .detect(data_bf, stride_list, sampling_rate_hz=sampling_rate_hz)
            .min_vel_event_list_
        )

        self.event_list_ = min_vel_stride_list

        # 2. We now have the strides defined as min_vel -> min_vel. We can integrate the data over these strides.
        # For the Kalman Filter we need to integrate over multiple strides to actually get the benefits of the Kalman
        # smoothing.
        # Hence, we create a fake roi list to pass as region to the trajectory reconstruction.
        #
        # However, in this dataset the recording often does not start with a resting period.
        # To make sure that the initial orientation guess is correct, we find a resting period before and after the
        # segmented strides and use them as starting and ending point for the trajectory reconstruction.
        int(np.round(self.search_region_padding_s * sampling_rate_hz))

        fake_roi_list = {}
        for i, sensor in enumerate(data):
            min_vel_list = min_vel_stride_list[sensor]
            # start = int(max(0, min_vel_list.iloc[0].start - padding_s))
            # end = int(min(min_vel_list.iloc[-1].end + padding_s, len(data[sensor])))
            # fake_stride_list = pd.DataFrame.from_records(
            #     [
            #         {"start": start, "end": int(min_vel_list.iloc[0].start), "s_id": 0},
            #         {"start": int(min_vel_list.iloc[-1].end), "end": end, "s_id": 1},
            #     ]
            # )
            #
            # # We use Rampp EventDetection to find the static moments within these fake strides.
            # # This is 100% hacky, but it works.
            # fake_region_min_vels = (
            #     RamppEventDetection(detect_only=("min_vel",), enforce_consistency=False)
            #     .detect(
            #         data_bf[sensor], fake_stride_list, sampling_rate_hz=sampling_rate_hz
            #     )
            #     .segmented_event_list_
            # )
            #
            # fake_roi_list[sensor] = pd.DataFrame(
            #     {
            #         "start": [fake_region_min_vels.iloc[0].min_vel],
            #         "end": [fake_region_min_vels.iloc[1].min_vel],
            #         "roi_id": i,
            #     }
            # )
            fake_roi_list[sensor] = pd.DataFrame(
                {
                    "start": [min_vel_list.iloc[0].start],
                    "end": [min_vel_list.iloc[-1].end],
                    "roi_id": i,
                }
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = RegionLevelTrajectory(
                ori_method=None,
                pos_method=None,
                trajectory_method=self.traj_method,
            ).estimate_intersect(
                data,
                regions_of_interest=fake_roi_list,
                stride_event_list=min_vel_stride_list,
                sampling_rate_hz=sampling_rate_hz,
            )

        self.position_ = traj.position_
        self.orientation_ = traj.orientation_

        # 3. We now have the trajectory for each stride. We can calculate the parameters.
        # We will ignore warnings here, because parameter estimation will throw them because of our incomplete
        # stride list.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters_ = (
                SpatialParameterCalculation()
                .calculate(
                    min_vel_stride_list,
                    traj.position_,
                    traj.orientation_,
                    sampling_rate_hz=sampling_rate_hz,
                )
                .parameters_
            )
        return self


if __name__ == "__main__":
    # challenge.run(
    #     DummyOptimize(
    #         pipeline=Entry(RtsKalman(), search_region_padding_s=2.0),
    #     )
    # )
    # save_run(
    #     challenge=challenge,
    #     entry_name=("gaitmap", "rts_kalman", "default"),
    #     custom_metadata={
    #         "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
    #         "references": [],
    #         "code_authors": [],
    #         "algorithm_authors": [],
    #         "implementation_link": "",
    #     },
    #     path=Path("../"),
    # )

    paras = ParameterGrid({"traj_method__zupt_detector__inactive_signal_threshold": np.linspace(30, 80, 5)})

    challenge.run(
        GridSearch(
            pipeline=Entry(RtsKalman()),
            parameter_grid=paras,
            scoring=lambda x, y: -challenge.final_scorer(x, y)["abs_error_mean"],
            return_optimized=True,
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "rts_kalman", "optimized"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )

    # Set pandas print width to unlimited
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 10)
    print(
        challenge.get_core_results()["cv_results"][
            [
                "test_per_stride__abs_error_mean",
                "test_per_stride__abs_error_std",
                "test_per_stride__error_mean",
                "test_per_stride__error_std",
                "test_per_stride__icc",
            ]
        ]
    )
