from typing import Dict, cast

import pandas as pd
from gaitmap.stride_segmentation.hmm import HmmStrideSegmentation, RothSegmentationHmm
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge,
    ChallengeDataset,
    SensorNames,
)
from joblib import Memory
from tpcp import OptimizablePipeline, OptiPara, cf, make_action_safe, make_optimize_safe
from tpcp.optimize import Optimize
from typing_extensions import Self

from gaitmap_algos.stride_segmentation.roth_hmm import apply_and_flatten, metadata


class Entry(OptimizablePipeline[ChallengeDataset]):
    segmentation_model: OptiPara[RothSegmentationHmm]

    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame]

    def __init__(self, segmentation_model: RothSegmentationHmm = cf(RothSegmentationHmm())):
        self.segmentation_model = segmentation_model

    @make_optimize_safe
    def self_optimize(self, dataset: ChallengeDataset, **_) -> Self:
        all_bf_data = apply_and_flatten(
            dataset,
            lambda datapoint: convert_to_fbf(
                Challenge.get_imu_data(datapoint),
                left_like="l",
                right_like="r",
            ).values(),
        )

        all_ground_truth_stride_borders = apply_and_flatten(
            dataset,
            lambda datapoint: challenge.get_reference_stride_list(datapoint).values(),
        )

        self.segmentation_model.self_optimize(
            all_bf_data,
            all_ground_truth_stride_borders,
            sampling_rate_hz=dataset.sampling_rate_hz,
        )

        return self

    @make_action_safe
    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(challenge.get_imu_data(datapoint), left_like="l", right_like="r")
        self.stride_list_ = cast(
            Dict[SensorNames, pd.DataFrame],
            HmmStrideSegmentation(self.segmentation_model)
            .segment(bf_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            .stride_list_,
        )
        return self


if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(Optimize(Entry()))
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "roth_hmm", "trained_default"),
        custom_metadata=metadata,
    )
