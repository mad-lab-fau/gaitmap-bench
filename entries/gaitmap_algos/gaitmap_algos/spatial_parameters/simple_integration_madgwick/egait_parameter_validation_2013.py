from pathlib import Path
from typing import Literal

from gaitmap.trajectory_reconstruction import (
    ForwardBackwardIntegration,
    MadgwickAHRS,
)
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_parameter_validation_2013 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from optuna import Trial, create_study
from tpcp.optimize.optuna import OptunaSearch

from gaitmap_algos.spatial_parameters._egait_parameter_validation_2013_base import (
    StrideLevelIntegrationEntry,
)

SensorNames = Literal["left_sensor", "right_sensor"]

dataset = ChallengeDataset(
    data_folder=Path("/home/arne/Documents/repos/work/datasets/eGaIT_database"),
    memory=Memory("../.cache"),
)

challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})


class Entry(StrideLevelIntegrationEntry):
    def get_challenge(self):
        # We overwrite this method here to bind a reference to the file global challenge object
        return challenge


if __name__ == "__main__":
    # challenge.run(
    #     DummyOptimize(
    #         pipeline=Entry(
    #             pos_method=ForwardBackwardIntegration(level_assumption=False),
    #             ori_method=MadgwickAHRS(),
    #         ),
    #     )
    # )

    # save_run(
    #     challenge=challenge,
    #     entry_name=("gaitmap", "simple_integration_madgwick", "default"),
    #     custom_metadata={
    #         "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
    #         "references": [],
    #         "code_authors": [],
    #         "algorithm_authors": [],
    #         "implementation_link": "",
    #     },
    #     path=Path("../"),
    # )
    def create_search_space(trial: Trial):
        trial.suggest_float("ori_method__beta", 0.0, 0.3)

    def get_study():
        return create_study(direction="minimize")

    c = challenge.run(
        OptunaSearch(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
            create_study=get_study,
            create_search_space=create_search_space,
            scoring=challenge.final_scorer,
            score_name="per_stride__abs_error_mean",
            show_progress_bar=True,
            n_trials=20,
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "optimized"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )
