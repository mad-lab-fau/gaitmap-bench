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
from optuna import Trial
from optuna.samplers import TPESampler
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.spatial_parameters._sensor_pos_stride_integration_base import (
    StrideIntegrationBase,
)
from gaitmap_algos.spatial_parameters.simple_integration_madgwick import default_metadata


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("ori_method__beta", 0.01, 0.2, log=True)


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    def get_study_params(seed: int):
        return {
            "direction": "minimize",
            "sampler": TPESampler(seed=seed),
        }

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        OptunaSearch(
            pipeline=StrideIntegrationBase(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
            get_study_params=get_study_params,
            create_search_space=optuna_search_space,
            scoring=lambda p, d: {"per_stride": challenge.get_scorer()(p, d)["per_stride"]},
            score_name="per_stride__abs_error_mean",
            n_trials=20,
            return_optimized=True,
            random_seed=42,
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "optimized"),
        custom_metadata=default_metadata,
    )
