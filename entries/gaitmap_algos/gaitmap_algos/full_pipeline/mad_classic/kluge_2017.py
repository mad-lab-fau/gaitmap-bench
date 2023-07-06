import pandas as pd
from gaitmap.event_detection import RamppEventDetection
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.stride_segmentation import (
    BarthOriginalTemplate,
    ConstrainedBarthDtw,
)
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.full_pipeline.kluge_2017 import Challenge, ChallengeDataset
from joblib import Memory
from tpcp import Pipeline, make_action_safe
from tpcp.optimize import DummyOptimize


class MadClassic(Pipeline[ChallengeDataset]):
    # Result objects
    gait_parameters_with_turns_: pd.DataFrame
    gait_parameters_: pd.DataFrame
    aggregated_gait_parameters_: pd.Series

    @make_action_safe
    def run(self, datapoint: ChallengeDataset):
        data_sf = Challenge.get_imu_data(datapoint)
        sampling_rate_hz = datapoint.sampling_rate_hz

        # preprocess
        bf_data = convert_to_fbf(data_sf, left_like="left_", right_like="right_")

        # stride segmentation
        dtw = ConstrainedBarthDtw(template=BarthOriginalTemplate(use_cols=["gyr_ml"]), max_cost=2.4)
        dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

        # event detection

        ed = RamppEventDetection()
        ed = ed.detect(
            data=bf_data,
            stride_list=dtw.stride_list_,
            sampling_rate_hz=sampling_rate_hz,
        )

        # trajectory estimation
        trajectory = StrideLevelTrajectory()
        trajectory = trajectory.estimate(
            data=data_sf,
            stride_event_list=ed.min_vel_event_list_,
            sampling_rate_hz=sampling_rate_hz,
        )

        # temporal parameters
        temporal_paras = TemporalParameterCalculation()
        temporal_paras = temporal_paras.calculate(
            stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz
        )

        spatial_paras = SpatialParameterCalculation()
        spatial_paras = spatial_paras.calculate(
            stride_event_list=ed.min_vel_event_list_,
            positions=trajectory.position_,
            orientations=trajectory.orientation_,
            sampling_rate_hz=sampling_rate_hz,
        )

        all_temporal = pd.concat(temporal_paras.parameters_)
        all_spatial = pd.concat(spatial_paras.parameters_)

        self.gait_parameters_with_turns_ = pd.concat([all_temporal, all_spatial], axis=1)
        self.gait_parameters_ = self.gait_parameters_with_turns_.query("turning_angle.abs() < 20")
        self.aggregated_gait_parameters_ = self.gait_parameters_.mean()

        return self


if __name__ == "__main__":
    metadata = {
        "short_description": "Default MaD pipeline",
        "long_description": "The classic pipeline used in many gait-analysis studies by the MaD lab. "
        "This version uses mostly the default parameters of the gaitmap-implementations without "
        "specific tuning.",
        "references": [
            "https://ieeexplore.ieee.org/document/6949634",
            "http://www.mdpi.com/1424-8220/15/3/6419",
        ],
        "code_authors": ["MaD-DiGait"],
        "algorithm_authors": ["See source and references for individual algorithm authors"],
        "implementation_url": "https://github.com/mad-lab-fau/gaitmap",
    }

    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(MadClassic()),
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "mad_classic", "default"),
        custom_metadata=metadata,
    )
