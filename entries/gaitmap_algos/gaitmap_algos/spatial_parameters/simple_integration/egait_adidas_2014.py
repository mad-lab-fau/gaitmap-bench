import warnings
from typing import Dict

import pandas as pd
from gaitmap.base import BaseOrientationMethod, BasePositionMethod
from gaitmap.event_detection import RamppEventDetection
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.trajectory_reconstruction import (
    ForwardBackwardIntegration,
    SimpleGyroIntegration,
    StrideLevelTrajectory,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_bench import set_config
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
    SensorNames,
)
from joblib import Memory
from tpcp import Pipeline
from tpcp.optimize import DummyOptimize
from typing_extensions import Self


class Entry(Pipeline[ChallengeDataset]):
    # Result object
    parameters_: Dict[SensorNames, pd.DataFrame]
    event_list_: Dict[SensorNames, pd.DataFrame]
    position_: Dict[SensorNames, pd.DataFrame]
    orientation_: Dict[SensorNames, pd.DataFrame]

    def __init__(
        self,
        ori_method: BaseOrientationMethod,
        pos_method: BasePositionMethod,
    ):
        self.ori_method = ori_method
        self.pos_method = pos_method

    def run(self, datapoint: ChallengeDataset) -> Self:
        data = Challenge.get_imu_data(datapoint)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_bf = convert_to_fbf(data, left_like="left_", right_like="right_")
        stride_list = Challenge.get_ground_truth_segmented_stride_list(datapoint)
        sampling_rate_hz = datapoint.sampling_rate_hz

        # 1. We only have segmented strides -> we need to find the min_vel for each stride. For this we use Rampp
        # Event Detection, but only calculate the min_vel.
        self.event_list_ = (
            RamppEventDetection(detect_only=("min_vel",))
            .detect(data_bf, stride_list, sampling_rate_hz=sampling_rate_hz)
            .min_vel_event_list_
        )

        # 2. We now have the strides defined as min_vel -> min_vel. We can integrate the data over these strides.

        traj = StrideLevelTrajectory(ori_method=self.ori_method, pos_method=self.pos_method).estimate(
            data, self.event_list_, sampling_rate_hz=sampling_rate_hz
        )

        self.position_ = traj.position_
        self.orientation_ = traj.orientation_

        # 3. We now have the trajectory for each stride. We can calculate the parameters.
        # We will ignore warnings here, because parameter estimation will throw them because of our incomplete
        # stride list.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.parameters_ = (
                SpatialParameterCalculation(calculate_only=("stride_length", "gait_velocity"))
                .calculate(
                    self.event_list_,
                    traj.position_,
                    traj.orientation_,
                    sampling_rate_hz=sampling_rate_hz,
                )
                .parameters_
            )
        return self


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=SimpleGyroIntegration(),
            ),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration", "default"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
    )
