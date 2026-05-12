# Plotting

This package contains stateless plotting helpers for experiment outputs.

- `colors.py`: shared color dictionaries and color helper functions
- `data.py`: summary tables and trajectory dataframe transforms
- `metrics.py`: metric overview and metric bar plots
- `trajectories.py`: loss-curve plots
- `ablations.py`: scenario and ablation metric plots
- `PlottingUsage.ipynb`: small synthetic example showing how to call the API

Plot functions take data as inputs and return Matplotlib figures and axes. They
do not read experiment files, mutate global experiment state, or save figures.
