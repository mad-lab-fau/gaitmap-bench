from typing import Dict, Literal

import pandas as pd
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
)
from tpcp import Pipeline, make_action_safe
from typing_extensions import Self

SensorNames = Literal["left_sensor", "right_sensor"]


class SensorPosDtwBase(Pipeline[ChallengeDataset]):
    dtw: BarthDtw

    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame]

    def __init__(self, dtw: BarthDtw):
        self.dtw = dtw

    @make_action_safe
    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(Challenge.get_imu_data(datapoint), left_like="l", right_like="r")
        self.stride_list_ = self.dtw.clone().segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz).stride_list_
        return self
