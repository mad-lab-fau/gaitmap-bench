from gaitmap.stride_segmentation import BarthDtw
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014_original_label import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial, create_study
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.stride_segmentation.dtw._egait_segmentation_validation_2014 import (
    Egait2014DtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.barth_dtw import metadata_optimized


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("dtw__max_cost", 2.0, 3.5)
    trial.suggest_categorical(
        "dtw__template__use_cols",
        ['("gyr_ml", "gyr_si", "gyr_pa")', '("gyr_ml", "gyr_si")', '("gyr_ml",)'],
    )


def get_study_params(_):
    return dict(direction="maximize")


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        OptunaSearch(
            pipeline=Egait2014DtwBase(dtw=BarthDtw()),
            get_study_params=get_study_params,
            scoring=challenge.get_scorer(),
            score_name="per_sample__f1_score",
            create_search_space=optuna_search_space,
            return_optimized=True,
            n_trials=100,
            eval_str_paras=["dtw__template__use_cols"],
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "barth_dtw", "optimized"),
        custom_metadata=metadata_optimized,
    )
