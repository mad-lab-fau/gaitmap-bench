from gaitmap.trajectory_reconstruction import (
    ForwardBackwardIntegration,
    MadgwickAHRS,
)
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.spatial_parameters.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._sensor_pos_stride_integration_base import (
    StrideIntegrationBase,
)
from gaitmap_algos.spatial_parameters.simple_integration_madgwick import default_metadata

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=StrideIntegrationBase(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "default"),
        custom_metadata=default_metadata,
    )
