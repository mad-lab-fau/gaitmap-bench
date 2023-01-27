from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict

import pandas as pd
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_algos.entries.stride_segmentation.roth_hmm._shared import metadata
from gaitmap_bench import save
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
)
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Segmentation,
)
from gaitmap_mad.stride_segmentation.hmm import HmmStrideSegmentation, RothSegmentationHmm
from joblib import Memory
from sklearn.model_selection import KFold
from tpcp import OptiPara, OptimizablePipeline
from tpcp.optimize import Optimize
from typing_extensions import Literal, Self

SensorNames = Literal["l_instep", "l_instep"]

dataset = SensorPositionComparison2019Segmentation(
    data_folder=Path(
        "/home/arne/Documents/repos/work/projects/sensor_position_comparison/sensor_position_main_analysis/data/raw/"
    ),
    memory=Memory("../.cache"),
)

challenge = Challenge(
    dataset=dataset, cv_iterator=KFold(3, shuffle=True), cv_params={"n_jobs": 3}
)


@dataclass(repr=False)
class Entry(OptimizablePipeline[SensorPositionComparison2019Segmentation]):

    segmentation_model: OptiPara[RothSegmentationHmm] = field(
        default_factory=lambda: RothSegmentationHmm().set_params(
            stride_model__max_iterations=1, transition_model__max_iterations=1
        )
    )

    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame] = field(init=False)

    def self_optimize(self, dataset: SensorPositionComparison2019Segmentation, **kwargs) -> Self:
        all_bf_data = list(
            chain(
                *(
                    list(
                        convert_to_fbf(
                            challenge.get_imu_data(datapoint), left_like="l", right_like="r"
                        ).values()
                    )
                    for datapoint in dataset
                )
            )
        )
        all_ground_truth_stride_borders = list(
            chain(
                *(
                    list(challenge.get_reference_stride_list(datapoint).values())
                    for datapoint in dataset
                )
            )
        )
        self.segmentation_model.self_optimize(
            all_bf_data,
            all_ground_truth_stride_borders,
            sampling_rate_hz=dataset.sampling_rate_hz,
        )

        return self

    def run(self, datapoint: SensorPositionComparison2019Segmentation) -> Self:
        bf_data = convert_to_fbf(
            challenge.get_imu_data(datapoint), left_like="l", right_like="r"
        )
        self.stride_list_ = (
            HmmStrideSegmentation(self.segmentation_model)
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_
        )
        return self
if __name__ == "__main__":
    challenge.run(Optimize(Entry()))
    save(metadata, challenge)