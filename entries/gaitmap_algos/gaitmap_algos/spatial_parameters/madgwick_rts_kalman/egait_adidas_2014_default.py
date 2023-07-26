from gaitmap.trajectory_reconstruction import MadgwickRtsKalman
from gaitmap_bench import set_config, save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._egait_adidas_region_integration_base import RegionIntegrationBase
from gaitmap_algos.spatial_parameters.madgwick_rts_kalman import default_metadata

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=RegionIntegrationBase(MadgwickRtsKalman()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "madgwick_rts_kalman", "default"),
        custom_metadata=default_metadata,
    )