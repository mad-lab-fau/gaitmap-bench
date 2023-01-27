from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, Union

import pandas as pd
from attr import define, field
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge, ChallengeDataset
)
from joblib import Memory
from optuna import Trial, create_study
from sklearn.model_selection import KFold
from tpcp import Pipeline
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import Scorer
from typing_extensions import Literal, Self
from ._shared import metadata


from gaitmap_bench import save

dataset = ChallengeDataset(
    data_folder=Path(
        "/home/arne/Documents/repos/work/projects/sensor_position_comparison/sensor_position_main_analysis/data/raw/"
    ),
    memory=Memory("../.cache"),
)

challenge = Challenge(
    dataset=dataset, cv_iterator=KFold(3, shuffle=True), cv_params={"n_jobs": 1}
)


@define(kw_only=True, slots=False, repr=False)
class Entry(Pipeline[ChallengeDataset]):
    max_cost: float = 4

    # Result objects
    stride_list_: Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame] = field(init=False)

    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(
            challenge.get_imu_data(datapoint), left_like="l", right_like="r"
        )
        self.stride_list_ = (
            BarthDtw(max_cost=self.max_cost, memory=Memory("../.cache"))
            .set_params(template__use_cols=["gyr_ml", "gyr_si"])
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_
        )
        return self


@define(kw_only=True, slots=False, repr=False)
class ParaSearch(
    CustomOptunaOptimize.as_attrs()[Entry, ChallengeDataset]
):
    create_search_space: Callable[[Trial], None]

    def create_objective(
        self,
    ) -> Callable[
        [Trial, Entry, ChallengeDataset],
        Union[float, Sequence[float]],
    ]:
        def objective(
            trial: Trial,
            pipeline: Entry,
            dataset: ChallengeDataset,
        ) -> float:
            # First we need to select parameters for the current trial
            self.create_search_space(trial)
            # Then we apply these parameters to the pipeline
            # We wrap the score function with a scorer to avoid writing our own for-loop to aggregate the results.
            scorer = Scorer(challenge.final_scorer)

            pipeline = pipeline.set_params(**trial.params)
            average_score, _ = scorer(pipeline, dataset)
            return average_score["per_sample__f1_score"]

        return objective


def optuna_search_space(trial: Trial) -> None:
    trial.suggest_float("max_cost", 2.5, 5)


def get_study():
    return create_study(direction="maximize")


challenge.run(
    ParaSearch(
        pipeline=Entry(),
        create_study=get_study,
        create_search_space=optuna_search_space,
        return_optimized=True,
        n_trials=1,
    )
)

save(metadata, challenge)
