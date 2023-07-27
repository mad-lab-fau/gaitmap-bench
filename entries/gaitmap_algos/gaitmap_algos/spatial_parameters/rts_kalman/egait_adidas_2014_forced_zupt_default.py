from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.zupt_detection import (
    ComboZuptDetector,
    NormZuptDetector,
    StrideEventZuptDetector,
)
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._egait_adidas_region_integration_base import RegionIntegrationBase
from gaitmap_algos.spatial_parameters.rts_kalman import improved_zupt_default_metadata

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

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
            pipeline=RegionIntegrationBase(RtsKalman(zupt_detector=zupt_detector)),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "rts_kalman_forced_zupt", "default"),
        custom_metadata=improved_zupt_default_metadata,
    )
