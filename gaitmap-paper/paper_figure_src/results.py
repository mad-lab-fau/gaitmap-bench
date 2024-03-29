from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from gaitmap_challenges.full_pipeline.kluge_2017 import Challenge as FullPipelineChallenge
from gaitmap_challenges.results import filter_results, get_all_result_paths, get_latest_result, load_run, rename_keys
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge as SegmentationChallenge
from gaitmap_challenges.visualization import SingleMetricBoxplot, group_by_data_label, replace_legend_labels

sns.set_palette(["#FFB81C", "#4C3B70", "#009B77", "#00A3E0"])
sns.set_context("paper", font_scale=0.75)

flierprops = {"marker": "o", "markersize": 3, "markerfacecolor": "None", "markeredgecolor": "black"}

HERE = Path(__file__).parent
RESULTS_FOLDER = Path(HERE.parent.parent, "results")
PAPER_FOLDER = Path("/home/arne/Documents/repos/work/gaitmap_paper/src/figures/plots")
TWO_COLUMN_WIDTH = 7.16  # in inches

# %%
all_runs = get_all_result_paths(SegmentationChallenge, RESULTS_FOLDER)
all_runs = filter_results(all_runs, challenge_version=SegmentationChallenge.VERSION, is_debug_run=False)
latest_runs = get_latest_result(all_runs)

rename_map = {
    ("gaitmap", "barth_dtw", "default"): "DTW\nDefault",
    ("gaitmap", "barth_dtw", "optimized"): "DTW\nOptimized",
    ("gaitmap", "constrained_barth_dtw", "default"): "Constrained DTW\nDefault",
    ("gaitmap", "constrained_barth_dtw", "optimized"): "Constrained DTW\nOptimized",
    ("gaitmap", "roth_hmm", "default"): "HMM\nDefault",
    ("gaitmap", "roth_hmm", "trained_default"): "HMM\nRe-Trained",
}

order = list(rename_map.values())


run_info = {k: load_run(SegmentationChallenge, v) for k, v in latest_runs.items()}
cv_results = rename_keys({k: v.results["cv_results"] for k, v in run_info.items()}, rename_map)

per_test = SingleMetricBoxplot(
    cv_results,
    "f1_score",
    "single",
    overlay_scatter=False,
    force_order=order,
    label_grouper=group_by_data_label(level="test", include_all="Combined"),
    matplotlib_boxplot_props={"flierprops": flierprops},
)


per_fold = SingleMetricBoxplot(
    cv_results,
    "f1_score",
    "fold",
    overlay_scatter=False,
    force_order=order,
    matplotlib_boxplot_props={"flierprops": flierprops},
)

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(TWO_COLUMN_WIDTH, 3))
per_test.matplotlib(ax=axs[0])
with sns.color_palette(["#009B77"] * 6):
    per_fold.matplotlib(ax=axs[1])

axs[0].set_ylabel("F1-Score")
axs[1].set_ylabel(None)

axs[0].set_title("Per Test")
axs[1].set_title("Per Fold")

for ax in axs:
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

sns.move_legend(axs[0], 0, title="Test")
replace_legend_labels(axs[0], {"free_walk": "Free Walk"})

fig.tight_layout()

fig.savefig(PAPER_FOLDER / "example_1_box.pdf")

# %%

rename_map = {("gaitmap", "mad_modern", "default"): "Modern", ("gaitmap", "mad_classic", "default"): "Classic"}

order = ["Classic", "Modern"]

all_runs = get_all_result_paths(FullPipelineChallenge, RESULTS_FOLDER)
all_runs = filter_results(all_runs, challenge_version=FullPipelineChallenge.VERSION, is_debug_run=False)
latest_runs = get_latest_result(all_runs)

run_info = {k: load_run(FullPipelineChallenge, v) for k, v in latest_runs.items()}
cv_results = rename_keys({k: v.results["cv_results"] for k, v in run_info.items()}, rename_map)


per_test = SingleMetricBoxplot(
    cv_results,
    "gait_velocity__abs_error",
    "single",
    force_order=order,
    overlay_scatter=False,
    label_grouper=group_by_data_label(
        level="speed", include_all="Combined", force_order=["slow", "normal", "fast", "Combined"]
    ),
    matplotlib_boxplot_props={"flierprops": flierprops},
)
per_fold = SingleMetricBoxplot(
    cv_results,
    "agg__gait_velocity__abs_error_mean",
    "fold",
    force_order=order,
    overlay_scatter=False,
    matplotlib_boxplot_props={"flierprops": flierprops},
)

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(TWO_COLUMN_WIDTH, 2.2))
per_test.matplotlib(ax=axs[0])
with sns.color_palette(["#009B77"] * 2):
    per_fold.matplotlib(ax=axs[1])

sns.move_legend(axs[0], 0, title="Speed")
replace_legend_labels(axs[0], {"slow": "Slow", "normal": "Normal", "fast": "Fast"})

axs[0].set_ylabel("Absolute Gait Speed Error [m/s]")
axs[1].set_ylabel(None)

axs[0].set_title("Per Test")
axs[1].set_title("Per Fold")

fig.tight_layout()

fig.savefig(PAPER_FOLDER / "example_2_box.pdf")
