from pathlib import Path

import pandas as pd
from gaitmap.trajectory_reconstruction import (
    ForwardBackwardIntegration,
    SimpleGyroIntegration,
)
from gaitmap_challenges import save_run
from gaitmap_challenges.spatial_parameters.egait_parameter_validation_2013 import (
    Challenge,
    ChallengeDataset,
)
from joblib import Memory
from tpcp.optimize import DummyOptimize

from gaitmap_algos.spatial_parameters._egait_parameter_validation_2013_base import (
    StrideLevelIntegrationEntry,
)

dataset = ChallengeDataset(
    data_folder=Path("/home/arne/Documents/repos/work/datasets/eGaIT_database"),
    memory=Memory("../.cache"),
)

challenge = Challenge(dataset=dataset, cv_params={"n_jobs": 3})


# class ForwardBackwardIntegrationDedrifted(ForwardBackwardIntegration):
#     def estimate(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
#         # We apply a linear drift correction to the acceleration data
#         # By creating a linear drift function from the first and last point using interpolation
#         plateau_start = np.round(0.04 * len(data)).astype(int)
#         plateau_end = np.round(0.02 * len(data)).astype(int)
#
#         data = data.copy()
#         data[["acc_x", "acc_y", "acc_z"]] -= [0, 0, 9.81]
#         for col in ["acc_x", "acc_y", "acc_z"]:
#             partial_data = data[col].iloc[plateau_start:-plateau_end]
#             start = data[col].iloc[:plateau_start].mean()
#             end = data[col].iloc[-plateau_end:].mean()
#             data.loc[data.index[:plateau_start], col] = (
#                 data[col].iloc[:plateau_start] - start
#             )
#             data.loc[data.index[-plateau_end:], col] = (
#                 data[col].iloc[-plateau_end:] - end
#             )
#             data.loc[partial_data.index, col] = partial_data - interp1d(
#                 [0, len(partial_data)], [start, end]
#             )(np.arange(len(partial_data)))
#
#         return super().estimate(data, sampling_rate_hz)


class Entry(StrideLevelIntegrationEntry):
    def get_challenge(self) -> Challenge:
        # We overwrite this method here to bind a reference to the file global challenge object
        return challenge


if __name__ == "__main__":
    challenge.run(
        DummyOptimize(
            pipeline=Entry(
                pos_method=ForwardBackwardIntegration(level_assumption=False),
                ori_method=SimpleGyroIntegration(),
            ),
        )
    )
    save_run(
        challenge=challenge,
        entry_name=("gaitmap", "simple_integration", "default"),
        custom_metadata={
            "description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
            "references": [],
            "code_authors": [],
            "algorithm_authors": [],
            "implementation_link": "",
        },
        path=Path("../"),
    )

    # Set pandas print width to unlimited
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", 10)
    print(
        challenge.get_core_results()["cv_results"][
            [
                "test_per_stride__abs_error_mean",
                "test_per_stride__abs_error_std",
                "test_per_stride__error_mean",
                "test_per_stride__error_std",
                "test_per_stride__icc",
            ]
        ]
    )
