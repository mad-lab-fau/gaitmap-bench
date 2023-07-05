import warnings
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from gaitmap.base import BaseOrientationMethod, BasePositionMethod
from gaitmap.event_detection import RamppEventDetection
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.trajectory_reconstruction import ForwardBackwardIntegration, MadgwickAHRS, StrideLevelTrajectory
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial, create_study
from tpcp import Pipeline
from tpcp.optimize.optuna import OptunaSearch
from typing_extensions import Self

dataset = ChallengeDataset(
    data_folder=Path(
        "/home/arne/Documents/repos/work/datasets/ValidationDatasets-GaitAnalysis/MoCapReference_Adidas/public_version/"
    ),
    memory=Memory("../.cache"),
).get_subset(sensor="shimmer3")

challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})

SensorNames = Literal["left_sensor", "right_sensor"]


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

    def get_challenge(self) -> Challenge:
        return challenge

    def run(self, datapoint: ChallengeDataset) -> Self:
        challenge = self.get_challenge()
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

        traj = StrideLevelTrajectory(ori_method=self.ori_method, pos_method=self.pos_method).estimate(
            data, min_vel_stride_list, sampling_rate_hz=sampling_rate_hz
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
    # print(Entry(
    #     pos_method=ForwardBackwardIntegration(level_assumption=False),
    #     ori_method=SimpleGyroIntegration(),
    # ).run(dataset[0]).parameters_)
    # paras = ParameterGrid({"ori_method__beta": np.linspace(0, 0.2, 5)})
    # c = challenge.run(
    #     GridSearch(
    #         pipeline=Entry(
    #             pos_method=ForwardBackwardIntegration(level_assumption=False),
    #             ori_method=MadgwickAHRS(),
    #         ),
    #         scoring=challenge.final_scorer,
    #         return_optimized="-per_stride__abs_error_mean",
    #         parameter_grid=paras,
    #     )
    # )

    def create_search_space(trial: Trial):
        trial.suggest_float("ori_method__beta", 0.0, 0.3)

    def get_study():
        return create_study(direction="minimize")

    c = challenge.run(
        OptunaSearch(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
            create_study=get_study,
            create_search_space=create_search_space,
            scoring=challenge.final_scorer,
            score_name="per_stride__abs_error_mean",
            n_trials=25,
        )
    )
    print(c.get_core_results()["opti_results"])
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "optimized"),
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
