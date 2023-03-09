from pathlib import Path
from pathlib import Path
from typing import Dict, cast

import pandas as pd
from gaitmap.stride_segmentation.hmm import (
    PreTrainedRothSegmentationModel,
    HmmStrideSegmentation,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from joblib import Memory
from sklearn.model_selection import KFold
from tpcp import Pipeline, make_action_safe
from tpcp.optimize import DummyOptimize
from typing_extensions import Self

from gaitmap_algos.stride_segmentation.roth_hmm import metadata
from gaitmap_bench import save_run
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge,
    ChallengeDataset,
    SensorNames,
)


class Entry(Pipeline[ChallengeDataset]):
    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame]

    @make_action_safe
    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(
            Challenge.get_imu_data(datapoint), left_like="l", right_like="r"
        )
        self.stride_list_ = cast(
            Dict[SensorNames, pd.DataFrame],
            HmmStrideSegmentation(PreTrainedRothSegmentationModel())
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_,
        )
        return self


if __name__ == "__main__":
    dataset = ChallengeDataset(
        data_folder=Path(
            "/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation"
        ),
        memory=Memory("../.cache"),
    )

    challenge = Challenge(
        dataset=dataset, cv_iterator=KFold(3, shuffle=True), cv_params={"n_jobs": 3}
    )

    challenge.run(DummyOptimize(Entry()))
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "roth_hmm", "default"),
        custom_metadata=metadata,
        path=Path("../"),
    )
