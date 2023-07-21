from gaitmap.trajectory_reconstruction import (
    RtsKalman,
)
from gaitmap_bench import set_config
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial, create_study
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.spatial_parameters._egait_adidas_region_integration_base import RegionIntegrationBase


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("traj_method__zupt_detector__inactive_signal_threshold", 30, 80)


def get_study():
    return create_study(direction="minimize")


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        OptunaSearch(
            pipeline=RegionIntegrationBase(RtsKalman()),
            create_study=get_study,
            create_search_space=optuna_search_space,
            scoring=challenge.get_scorer(),
            score_name="abs_error_mean",
            n_trials=10,
            return_optimized=True,
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "rts_kalman", "optimized"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
    )
