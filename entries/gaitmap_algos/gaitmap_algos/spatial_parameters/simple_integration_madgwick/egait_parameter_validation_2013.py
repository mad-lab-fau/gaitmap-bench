from pathlib import Path
from typing import Literal

import numpy as np
from gaitmap.trajectory_reconstruction import (
    MadgwickAHRS,
    ForwardBackwardIntegration,
)
from gaitmap_algos.spatial_parameters._egait_parameter_validation_2013_base import (
    StrideLevelIntegrationEntry,
)
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_parameter_validation_2013 import (
    ChallengeDataset,
    Challenge,
)
from joblib import Memory
from sklearn.model_selection import ParameterGrid
from tpcp.optimize import GridSearch, DummyOptimize

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
    challenge.run(
        DummyOptimize(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
        )
    )

    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "default"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "citations": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )
    paras = ParameterGrid({"ori_method__beta": np.linspace(0, 0.1, 10)})

    challenge.run(
        GridSearch(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=MadgwickAHRS(),
            ),
            parameter_grid=paras,
            scoring=lambda x, y: -challenge.final_scorer(x, y)["abs_error_mean"],
            return_optimized=True,
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration_madgwick", "optimized"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "citations": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )
