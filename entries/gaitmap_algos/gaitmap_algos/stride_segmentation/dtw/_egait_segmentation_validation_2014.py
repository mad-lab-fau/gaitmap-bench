from typing import Dict

import pandas as pd
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from tpcp import Pipeline, make_action_safe
from typing_extensions import Literal, Self

from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge,
    ChallengeDataset,
)

SensorNames = Literal["left_sensor", "right_sensor"]


class Egait2014DtwBase(Pipeline[ChallengeDataset]):
    dtw: BarthDtw

    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame]

    def __init__(self, dtw: BarthDtw):
        self.dtw = dtw

    @make_action_safe
    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(
            Challenge.get_imu_data(datapoint), left_like="l", right_like="r"
        )
        self.stride_list_ = (
            self.dtw.clone()
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_
        )
        return self
