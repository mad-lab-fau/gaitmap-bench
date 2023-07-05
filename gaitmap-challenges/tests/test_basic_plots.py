from pathlib import Path

from gaitmap_challenges.results import load_run, load_run_metadata
from gaitmap_challenges.visualization._basic_plots import box_plot_bokeh, group_by_data_label

HERE = Path(__file__).parent


def get_all_result_path(challenge_class, base_path):
    # TODO: Handle challenge version!

    challenge_class = challenge_class if isinstance(challenge_class, type) else challenge_class.__class__
    folder_name = challenge_class.__module__ + "." + challenge_class.__name__
    try:
        folder = next(Path(base_path).rglob(folder_name))
    except StopIteration as e:
        raise FileNotFoundError(f"Could not find any results for {challenge_class.__name__}") from e

    entries = {}
    for run in folder.rglob("metadata.json"):
        meta = load_run_metadata(run.parent)
        entries.setdefault(tuple(meta["entry_name"]), []).append(run.parent)

    sorted_entries = {}
    for name, entry_list in entries.items():
        parents = {e.parent for e in entry_list}
        if len(parents) > 1:
            # TODO: Add a warning. All results of a challenge should be in the same folder. Otherwise someone might
            #  have manully copied them around
            pass
        sorted_entries[name] = sorted(entry_list, key=lambda e: e.name, reverse=False)

    return entries


class TestBokehBoxplot:
    def test_grouped(self):
        from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge

        all_runs = get_all_result_path(Challenge, HERE.parent.parent / "results")

        run_info = {k: load_run(Challenge, v[0]) for k, v in all_runs.items()}

        cv_results = {k: v.results["cv_results"] for k, v in run_info.items()}

        box_plot_bokeh(cv_results, "precision", "single", overlay_scatter=True, label_grouper=group_by_data_label(1))
