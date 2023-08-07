from pathlib import Path

from gaitmap_challenges.results import load_run, get_all_results_paths
from gaitmap_challenges.visualization import box_plot_bokeh, group_by_data_label

HERE = Path(__file__).parent


class TestBokehBoxplot:
    def test_grouped(self):
        from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge

        all_runs = get_all_result_paths(Challenge, HERE.parent.parent / "results")

        run_info = {k: load_run(Challenge, v[0]) for k, v in all_runs.items()}

        cv_results = {k: v.results["cv_results"] for k, v in run_info.items()}

        box_plot_bokeh(cv_results, "precision", "single", overlay_scatter=True, label_grouper=group_by_data_label(1))
