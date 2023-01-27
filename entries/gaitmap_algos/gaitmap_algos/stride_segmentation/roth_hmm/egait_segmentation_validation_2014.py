from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd
from gaitmap.stride_segmentation.hmm import (
    PreTrainedRothSegmentationModel,
    HmmStrideSegmentation,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_algos.entries.stride_segmentation.roth_hmm._shared import metadata
from gaitmap_bench import save
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge, ChallengeDataset
)
from joblib import Memory
from sklearn.model_selection import KFold
from tpcp import make_action_safe
from tpcp.optimize import DummyOptimize
from typing_extensions import Literal, Self

SensorNames = Literal["left_sensor", "right_sensor"]

dataset = ChallengeDataset(
    data_folder=Path(
        "/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation"
    ),
    memory=Memory("../.cache"),
)

challenge = Challenge(
    dataset=dataset, cv_iterator=KFold(3, shuffle=True), cv_params={"n_jobs": 3}
)


@dataclass(repr=False)
class Entry(PipelineInterface):
    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame] = field(init=False)

    @make_action_safe
    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(challenge.get_imu_data(datapoint), left_like="l", right_like="r")
        self.stride_list_ = (
            HmmStrideSegmentation(PreTrainedRothSegmentationModel())
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_
        )
        return self


metadata = {**metadata, "name": "roth_hmm_pretrained"}


if __name__ == "__main__":
    challenge.run(DummyOptimize(Entry()))
    save(metadata, challenge)
