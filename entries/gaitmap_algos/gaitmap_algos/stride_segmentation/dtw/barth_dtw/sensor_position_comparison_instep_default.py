from gaitmap.stride_segmentation import BarthDtw
from gaitmap_bench import save_run, set_config
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.stride_segmentation.dtw._sensor_position_comparison_instep import (
    SensorPosDtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.barth_dtw import metadata

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=SensorPosDtwBase(dtw=BarthDtw()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "barth_dtw", "default"),
        custom_metadata=metadata,
    )
