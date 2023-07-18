from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    Challenge as BaseChallenge,
)
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import (
    ChallengeDataset,
    ResultType,
    SensorNames,
)


class Challenge(BaseChallenge):
    use_original_labels = True


__all__ = ["Challenge", "ChallengeDataset", "ResultType", "SensorNames"]
