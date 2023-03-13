from gaitmap.stride_segmentation import ConstrainedBarthDtw
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.stride_segmentation.dtw._egait_segmentation_validation_2014 import (
    Egait2014DtwBase,
)
from gaitmap_algos.stride_segmentation.dtw.constrained_barth_dtw import metadata
from gaitmap_bench import set_config, save_run
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge,
    ChallengeDataset,
)

if __name__ == "__main__":
    config = set_config()

    dataset = ChallengeDataset(
        memory=Memory(config.cache_dir),
    )

    challenge = Challenge(dataset=dataset, cv_params={"n_jobs": config.n_jobs})

    challenge.run(
        DummyOptimize(
            pipeline=Egait2014DtwBase(dtw=ConstrainedBarthDtw()),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "constrained_barth_dtw", "default"),
        custom_metadata=metadata,
    )
