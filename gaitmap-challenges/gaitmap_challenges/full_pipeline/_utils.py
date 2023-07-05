from typing import Dict, Literal, Sequence

import pandas as pd
from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors
from tpcp.validate import Aggregator

_AggType = Dict[Literal["reference", "predicted"], pd.Series]


class ParameterErrors(Aggregator[_AggType]):
    RETURN_RAW_SCORES = False

    @classmethod
    def aggregate(cls, /, values: Sequence[_AggType], datapoints) -> Dict[str, float]:
        all_references = pd.DataFrame({str(dp.group): val["reference"] for dp, val in zip(datapoints, values)}).T
        all_predictions = pd.DataFrame({str(dp.group): val["predicted"] for dp, val in zip(datapoints, values)}).T

        all_references.index.name = "group"
        all_predictions.index.name = "group"

        errors = calculate_aggregated_parameter_errors(
            reference_parameter=all_references,
            predicted_parameter=all_predictions,
            calculate_per_sensor=True,
            scoring_errors="ignore",
            id_column="group",
        ).stack()
        errors.index = [f"{c[1]}__{c[0]}" for c in errors.index]

        return errors.to_dict()
