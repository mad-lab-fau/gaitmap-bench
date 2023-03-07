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
from bokeh.plotting import figure
from myst_nb_bokeh import glue_bokeh

p = figure(width=300, height=300)
p.circle(list(range(1, 10)), list(range(10, 1, -1)));

glue_bokeh("bokeh_plot_1", p)
glue_bokeh("bokeh_plot_2", p)

```


`````{tab-set}
````{tab-item} Tab 1 title
```{glue:figure} bokeh_plot_2
:name: Results 1
```
````

````{tab-item} Tab 2 title
```{glue:figure} bokeh_plot_2
:name: Results 2
```
````
`````
