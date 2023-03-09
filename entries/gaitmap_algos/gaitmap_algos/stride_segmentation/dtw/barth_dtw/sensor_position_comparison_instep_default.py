from pathlib import Path

from gaitmap.stride_segmentation import BarthDtw
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.stride_segmentation.dtw._sensor_position_comparison_instep import (
    SensorPosDtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.barth_dtw import metadata
from gaitmap_challenges import save_run
from gaitmap_challenges.stride_segmentation.sensor_position_comparison_instep import (
    Challenge,
    ChallengeDataset,
)

if __name__ == "__main__":
    dataset = ChallengeDataset(
        data_folder=Path(
            "/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation"
        ),
        memory=Memory("../.cache"),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})

    challenge.run(
        DummyOptimize(
            pipeline=SensorPosDtwBase(dtw=BarthDtw()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "barth_dtw", "default"),
        custom_metadata=metadata,
        path=Path("../"),
    )
