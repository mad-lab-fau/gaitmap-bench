from pathlib import Path

from gaitmap.stride_segmentation import ConstrainedBarthDtw
from joblib import Memory
from optuna import Trial, create_study
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.stride_segmentation.dtw._sensor_position_comparison_instep import (
    SensorPosDtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.constrained_barth_dtw import metadata
from gaitmap_challenges import save_run
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
)


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("dtw__max_cost", 2.0, 3.5)
    trial.suggest_float("dtw__max_template_stretch_ms", 100.0, 300.0)
    trial.suggest_categorical(
        "dtw__template__use_cols",
        ['("gyr_ml", "gyr_si", "gyr_pa")', '("gyr_ml", "gyr_si")', '("gyr_ml",)'],
    )


def get_study():
    return create_study(direction="maximize")


if __name__ == "__main__":
    dataset = ChallengeDataset(
        data_folder=Path(
            "/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation"
        ),
        memory=Memory("../.cache"),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})

    challenge.run(
        OptunaSearch(
            pipeline=SensorPosDtwBase(dtw=ConstrainedBarthDtw()),
            create_study=get_study,
            scoring=challenge.get_scorer(),
            score_name="per_sample__f1_score",
            create_search_space=optuna_search_space,
            return_optimized=True,
            n_trials=10,
            eval_str_paras=["dtw__template__use_cols"],
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "constrained_barth_dtw", "optimized"),
        custom_metadata=metadata,
        path=Path("../"),
    )
