from typing import Dict, cast

import pandas as pd
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
    SensorNames,
)
from gaitmap_mad.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
)
from joblib import Memory
from tpcp import Pipeline
from tpcp.optimize import DummyOptimize
from typing_extensions import Self

from gaitmap_algos.stride_segmentation.roth_hmm import default_metadata


class Entry(Pipeline[ChallengeDataset]):
    # Result objects
    stride_list_: Dict[SensorNames, pd.DataFrame]

    def run(self, datapoint: ChallengeDataset) -> Self:
        bf_data = convert_to_fbf(challenge.get_imu_data(datapoint), left_like="l", right_like="r")
        self.stride_list_ = cast(
            Dict[SensorNames, pd.DataFrame],
            HmmStrideSegmentation(PreTrainedRothSegmentationModel())
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

    challenge.run(DummyOptimize(Entry()))
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "roth_hmm", "default"),
        custom_metadata=default_metadata,
    )
