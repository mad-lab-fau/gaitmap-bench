from pathlib import Path

from gaitmap.stride_segmentation import ConstrainedBarthDtw
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.stride_segmentation.dtw._egait_segmentation_validation_2014 import (
    Egait2014DtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.constrained_barth_dtw import metadata
from gaitmap_challenges import save_run
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
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
            pipeline=Egait2014DtwBase(dtw=ConstrainedBarthDtw()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "constrained_barth_dtw", "default"),
        custom_metadata=metadata,
        path=Path("../"),
    )
