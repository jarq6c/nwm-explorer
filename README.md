# National Water Model Evaluation Explorer

A web-based application used to explore National Water Model output and evaluation metrics. This package includes a command-line interface (CLI) for data retrieval and analysis, as well as a graphical user interface (GUI) for exploring evaluation results. The primary intended use-case is generating ad-hoc evaluations of National Water Model forecasts and analyses.

## Installation
```bash
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install -U pip wheel
(env) $ pip install nwm_explorer
```

## Command-line Interface
Once installed, the CLI is accessible from an activated python environment using `nwm-explorer`. For example,
```bash
$ nwm-explorer --help
```
```console
Usage: nwm-explorer [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build     Download and process data required by evaluations.
  display   Visualize and explore evaluation data.
  evaluate  Run a standard evaluation.
  export    Export predictions, observations, or evaluations to CSV.
```
Note that each command (`build`, `display`, `evaluate`, `export` will show additional information using `--help`)

### Standard Usage
Generally, users will want to run the `build`, `evaluate` and `display` commands in sequence to generate and explore NWM evaluations. Suppose we wanted to perform an ad-hoc evaluation of NWM forecasts issued from 2023-10-01 to 2023-10-03. We would run the following operations to achieve this:
```bash
# First, retrieve and pair the required data
# This command will retrieve model output and matching observations.
# It will use up to 4 cores (j) to for data processing and retry retrievals up to twice (r).
$ nwm-explorer build -s 2023-10-01 -e 2023-10-03 -j 4 -r 2

# Second, run the standard evaluation over the same period.
# Note here we give this evaluation a special label (l). If a label isn't specified,
# The software will assign a generic label.
$ nwm-explorer evaluate -s 2023-10-01 -e 2023-10-03 -j 4 -l my_evaluation

# Lastly, we can view the results of this evaluation using the GUI
$ nwm-explorer display
```

## Graphical User Interface

The GUI includes many options for exploring evaluation results including mapping of metrics, filtering by lead time or confidence bounds, regional histograms, hydrographs, and site information.

![GUI](https://raw.githubusercontent.com/jarq6c/nwm-explorer/main/images/gui.JPG)
