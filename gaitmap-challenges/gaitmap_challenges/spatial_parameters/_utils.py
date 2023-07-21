from typing import Dict, Literal, Sequence, Tuple

import pandas as pd
from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors
from tpcp.validate import Aggregator

_AggType = Tuple[
    str, Dict[Literal["reference", "predicted"], Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]]
]


class SingleValueErrors(Aggregator[_AggType]):
    RETURN_RAW_SCORES = False

    @classmethod
    def aggregate(cls, /, values: Sequence[_AggType], datapoints) -> Dict[str, float]:
        parameter = values[0][0]
        all_references = {}
        all_predictions = {}
        for i, (_, val) in enumerate(values):
            ref = val["reference"]
            pred = val["predicted"]
            for sensor in ["left_sensor", "right_sensor"]:
                try:
                    all_references[f"{i}_{sensor}"] = ref[sensor][[parameter]]
                    all_predictions[f"{i}_{sensor}"] = pred[sensor][[parameter]]
                except KeyError:
                    # Could be that the sensor is not available for this data
                    pass

        return calculate_aggregated_parameter_errors(
            reference_parameter=all_references,
            predicted_parameter=all_predictions,
            calculate_per_sensor=False,
            scoring_errors="ignore",
        )[parameter].to_dict()
