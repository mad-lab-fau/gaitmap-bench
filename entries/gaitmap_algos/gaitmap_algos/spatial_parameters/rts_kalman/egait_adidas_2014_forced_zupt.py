from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.zupt_detection import (
    ComboZuptDetector,
    ShoeZuptDetector,
    StrideEventZuptDetector,
)
from gaitmap_bench import set_config
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_adidas_2014 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial
from optuna.samplers import TPESampler
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.spatial_parameters._egait_adidas_region_integration_base import (
    RegionIntegrationBase,
)
from gaitmap_algos.spatial_parameters.rts_kalman import improved_zupt_default_metadata


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float(
        "traj_method__zupt_detector__detectors__shoe__inactive_signal_threshold",
        1e8,
        1e12,
        log=True,
    )
    trial.suggest_float(
        "traj_method__zupt_detector__detectors__shoe__acc_noise_variance",
        1e-9,
        1e-6,
        log=True,
    )
    trial.suggest_float(
        "traj_method__zupt_detector__detectors__shoe__gyr_noise_variance",
        1e-9,
        1e-6,
        log=True,
    )


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

    zupt_detector = ComboZuptDetector(
        [
            (
                "shoe",
                ShoeZuptDetector(),
            ),
            ("strides", StrideEventZuptDetector(half_region_size_s=0.025)),
        ]
    )

    challenge.run(
        OptunaSearch(
            pipeline=RegionIntegrationBase(RtsKalman(zupt_detector=zupt_detector)),
            get_study_params=get_study_params,
            create_search_space=optuna_search_space,
            scoring=lambda p, d: {"per_stride": challenge.get_scorer()(p, d)["per_stride"]},
            score_name="per_stride__abs_error_mean",
            n_trials=100,
            return_optimized=True,
        )
    )

    metadata = {
        **improved_zupt_default_metadata,
        "long_description": f"{improved_zupt_default_metadata['long_description']} "
        "To further improve the results the ZUPT threshold, the accelerometer noise variance and the gyroscope noise "
        "variance were optimized using a TPE Sampler.",
    }

    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "rts_kalman_forced_zupt", "optimized"),
        custom_metadata=improved_zupt_default_metadata,
    )
