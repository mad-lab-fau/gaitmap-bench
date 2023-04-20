---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: gaitmap_bench
---

# EgaitSegmentationValidation2014
```{code-cell} ipython3
:tags: [hide-input]
from pathlib import Path
from myst_nb_bokeh import glue_bokeh
from gaitmap_challenges.visualization._basic_plots import SingleMetricBoxplot, group_by_data_label
from gaitmap_challenges.results import load_run, load_run_metadata, get_latest_result, filter_results, get_all_results_path, generate_overview_table

from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge

from gaitmap_bench._config import DEFAULT_RESULTS_DIR


all_runs = get_all_results_path(Challenge, DEFAULT_RESULTS_DIR)
all_runs = filter_results(all_runs, challenge_version=Challenge.VERSION)
latest_runs = get_latest_result(all_runs)
generate_overview_table(latest_runs)
```

## Results per Participant and Test
```{code-cell} ipython3
:tags: [hide-input]
run_info = {k: load_run(Challenge, v) for k, v in latest_runs.items()}
cv_results = {k: v.results["cv_results"]  for k, v in run_info.items()}

for metric in ["f1_score", "precision", "recall"]:
    p = SingleMetricBoxplot(cv_results, metric, "single", overlay_scatter=True, label_grouper=group_by_data_label(level=1))
    glue_bokeh(f"single_{metric}", p.bokeh()) 
```


`````{tab-set}
:class: full-width
````{tab-item} F1-Score
```{glue:figure} single_f1_score
```
````
````{tab-item} Precision
```{glue:figure} single_precision
```
````
````{tab-item} Recall
```{glue:figure} single_recall
```
````
`````

## Results per CV Fold

```{code-cell} ipython3
:tags: [hide-input]
for metric in ["f1_score", "precision", "recall"]:
    p = SingleMetricBoxplot(cv_results, metric, "fold", overlay_scatter=True)
    glue_bokeh(f"fold_{metric}", p.bokeh())
```


`````{tab-set}
:class: full-width
````{tab-item} F1-Score
```{glue:figure} fold_f1_score
```
````
````{tab-item} Precision
```{glue:figure} fold_precision
```
````
````{tab-item} Recall
```{glue:figure} fold_recall
```
````
`````