from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_algos.entries.stride_segmentation.roth_hmm._shared import metadata
from gaitmap_bench import save
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge, ChallengeDataset
)

from gaitmap_mad.stride_segmentation.hmm import HmmStrideSegmentation, PreTrainedRothSegmentationModel
from joblib import Memory
from sklearn.model_selection import KFold
from tpcp import Pipeline
from tpcp.optimize import DummyOptimize
from typing_extensions import Literal, Self

SensorNames = Literal["l_instep", "l_instep"]

dataset = ChallengeDataset(
    data_folder=Path(
        "/home/arne/Documents/repos/work/projects/sensor_position_comparison/sensor_position_main_analysis/data/raw/"
    ),
    memory=Memory("../.cache"),
)

challenge = Challenge(
    dataset=dataset, cv_iterator=KFold(3, shuffle=True), cv_params={"n_jobs": 3}
)


@dataclass(repr=False)
class Entry(Pipeline[ChallengeDataset]):
    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame] = field(init=False)

    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(
            challenge.get_imu_data(datapoint), left_like="l", right_like="r"
        )
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