import warnings
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from gaitmap.base import BaseTrajectoryMethod
from gaitmap.event_detection import RamppEventDetection
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.trajectory_reconstruction import (
    MadgwickRtsKalman,
    RegionLevelTrajectory,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from gaitmap.zupt_detection import (
    ComboZuptDetector,
    NormZuptDetector,
    StrideEventZuptDetector,
)
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp import Pipeline
from tpcp.optimize import DummyOptimize
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
    ):
        self.traj_method = traj_method

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
        self.position_ = {}
        self.orientation_ = {}
        fake_roi_list = {}

        static_region_detector = NormZuptDetector(metric="variance")

        for i, sensor in enumerate(data):
            first_stride_start = int(min_vel_stride_list[sensor].iloc[0].start)
            last_stride_end = int(min_vel_stride_list[sensor].iloc[-1].end)
            fake_roi_list[sensor] = pd.DataFrame(
                {
                    "start": [
                        static_region_detector.detect(
                            data[sensor].iloc[:first_stride_start],
                            sampling_rate_hz=sampling_rate_hz,
                        ).min_vel_index_
                    ],
                    "end": [
                        static_region_detector.detect(
                            data[sensor].iloc[last_stride_end:],
                            sampling_rate_hz=sampling_rate_hz,
                        ).min_vel_index_
                        + last_stride_end
                    ],
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
                    self.position_,
                    self.orientation_,
                    sampling_rate_hz=sampling_rate_hz,
                )
                .parameters_
            )
        return self


if __name__ == "__main__":
    zupt_detector = ComboZuptDetector(
        [
            (
                "norm",
                NormZuptDetector(
                    sensor="gyr",
                    window_length_s=0.05,
                    window_overlap=0.5,
                    metric="maximum",
                    inactive_signal_threshold=55,
                ),
            ),
            ("strides", StrideEventZuptDetector(half_region_size_s=0.025)),
        ]
    )
    challenge.run(
        DummyOptimize(
            pipeline=Entry(MadgwickRtsKalman(zupt_detector=zupt_detector, madgwick_beta=0.1)),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "madgwick_rts_kalman", "forced_zupt"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )

    # zupt_detector = ComboZuptDetector(
    #     [
    #         (
    #             "norm",
    #             NormZuptDetector(
    #                 sensor="gyr",
    #                 window_length_s=0.05,
    #                 window_overlap=0.5,
    #                 metric="maximum",
    #                 inactive_signal_threshold=55,
    #             ),
    #         ),
    #         ("strides", StrideEventZuptDetector(half_region_size_s=0.025)),
    #     ]
    # )
    # challenge.run(
    #     DummyOptimize(
    #         pipeline=Entry(MadgwickRtsKalman(zupt_detector=zupt_detector, madgwick_beta=0.1)),
    #     )
    # )
    # save_run(
    #     challenge=challenge,
    #     entry_name=("gaitmap", "madgwick_rts_kalman", "forced_zupt_optimized"),
    #     custom_metadata={
    #         "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
    #         "references": [],
    #         "code_authors": [],
    #         "algorithm_authors": [],
    #         "implementation_link": "",
    #     },
    #     path=Path("../"),
    # )

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
