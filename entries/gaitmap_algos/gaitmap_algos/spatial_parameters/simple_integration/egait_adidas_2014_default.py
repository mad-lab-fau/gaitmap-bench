from gaitmap.trajectory_reconstruction import (
    ForwardBackwardIntegration,
    SimpleGyroIntegration,
)
from gaitmap_bench import set_config, save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._egait_adidas_stride_integration_base import (
    StrideIntegrationBase,
)
from gaitmap_algos.spatial_parameters.simple_integration import default_metadata

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
                ori_method=SimpleGyroIntegration(),
            ),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration", "default"),
        custom_metadata=default_metadata,
    )
