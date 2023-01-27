from typing import Dict, Sequence

import pandas as pd
from gaitmap.evaluation_utils import precision_recall_f1_score
from tpcp.validate import Aggregator


class SingleValuePrecisionRecallF1(Aggregator[pd.DataFrame]):
    @classmethod
    def aggregate(cls, /, values: Sequence[pd.DataFrame], datapoints) -> Dict[str, float]:
        return precision_recall_f1_score(pd.concat(values))
