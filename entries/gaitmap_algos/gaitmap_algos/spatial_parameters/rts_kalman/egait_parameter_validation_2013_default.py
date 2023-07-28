from gaitmap.trajectory_reconstruction import (
    RtsKalman,
)
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.spatial_parameters.egait_parameter_validation_2013 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._egait_parameter_validation_2013_region_integration_base import (
    RegionIntegrationBase,
)
from gaitmap_algos.spatial_parameters.rts_kalman import default_metadata

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=RegionIntegrationBase(RtsKalman()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "rts_kalman", "default"),
        custom_metadata=default_metadata,
    )
