import pandas as pd
from gaitmap.event_detection import HerzerEventDetection
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.trajectory_reconstruction import (
    MadgwickRtsKalman,
    RegionLevelTrajectory,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.zupt_detection import (
    ComboZuptDetector,
    NormZuptDetector,
    StrideEventZuptDetector,
)
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.full_pipeline.kluge_2017 import Challenge, ChallengeDataset
from gaitmap_mad.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
)
from joblib import Memory
from tpcp import Pipeline, make_action_safe
from tpcp.optimize import DummyOptimize

from gaitmap_algos.full_pipeline.mad_modern import shared_metadata


class MadOptimized(Pipeline[ChallengeDataset]):
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
        hmm = HmmStrideSegmentation(PreTrainedRothSegmentationModel()).segment(
            bf_data, sampling_rate_hz=datapoint.sampling_rate_hz
        )

        # event detection
        ed = HerzerEventDetection()
        ed = ed.detect(
            data=bf_data,
            stride_list=hmm.stride_list_,
            sampling_rate_hz=sampling_rate_hz,
        )

        # trajectory estimation
        fake_roi_list = {
            k: pd.DataFrame(
                {
                    "start": v.iloc[0].start,
                    "end": v.iloc[-1].end,
                    "roi_id": 1,
                },
                index=[0],
            )
            for k, v in ed.min_vel_event_list_.items()
        }
        trajectory = RegionLevelTrajectory(
            ori_method=None,
            pos_method=None,
            trajectory_method=MadgwickRtsKalman(
                zupt_detector=ComboZuptDetector(
                    detectors=[
                        ("stride", StrideEventZuptDetector(half_region_size_s=0.05)),
                        ("norm", NormZuptDetector()),
                    ]
                )
            ),
        )
        trajectory = trajectory.estimate_intersect(
            data=data_sf,
            stride_event_list=ed.min_vel_event_list_,
            regions_of_interest=fake_roi_list,
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
        "short_description": "Modern MaD pipeline with optimized ZUPT detection",
        "long_description": "A version of the modern MaD pipeline with more advanced ZUPT detection. Specfically, this"
        "method forces the existence of one ZUPT event per stride. "
        "This can help with fast walks, where a simple threshold based ZUPT detector tuned for "
        "normal speeds might not be able to detect ZUPT events.",
        **shared_metadata,
    }

    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(MadOptimized()),
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "mad_modern", "optimized_zupt"),
        custom_metadata=metadata,
    )
