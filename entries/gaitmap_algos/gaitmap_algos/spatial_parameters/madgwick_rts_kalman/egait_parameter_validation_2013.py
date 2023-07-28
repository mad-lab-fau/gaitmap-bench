from gaitmap.trajectory_reconstruction import MadgwickRtsKalman
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.spatial_parameters.egait_parameter_validation_2013 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial
from optuna.samplers import TPESampler
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.spatial_parameters._egait_parameter_validation_2013_region_integration_base import (
    RegionIntegrationBase,
)
from gaitmap_algos.spatial_parameters.madgwick_rts_kalman import (
    default_metadata,
    improved_zupt_default_metadata,
)


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("traj_method__madgwick_beta", 0.01, 0.2, log=True)


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(memory=Memory(config.cache_dir))

    dataset = dataset

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    def get_study_params(seed: int):
        return {
            "direction": "minimize",
            "sampler": TPESampler(seed=seed),
        }

    challenge.run(
        OptunaSearch(
            pipeline=RegionIntegrationBase(MadgwickRtsKalman()),
            get_study_params=get_study_params,
            create_search_space=optuna_search_space,
            scoring=lambda p, d: {"per_stride": challenge.get_scorer()(p, d)["per_stride"]},
            score_name="per_stride__abs_error_mean",
            n_trials=100,
            return_optimized=True,
            random_seed=42,
        )
    )

    metadata = {
        **improved_zupt_default_metadata,
        "long_description": f"{default_metadata['long_description']} "
        "To further improve the results the Madgwick beta parameter were optimized using a TPE-Sampler.",
    }

    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "madgwick_rts_kalman", "optimized"),
        custom_metadata=improved_zupt_default_metadata,
    )
