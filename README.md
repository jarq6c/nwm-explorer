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
                                                                                                                                                   
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                         │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                  │
│ --help                        Show this message and exit.                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ download   Download and process NWM and USGS data for evaluations.                                                                              │
│ pair       Resample and pair NWM predictions to USGS observations.                                                                              │
│ compute    Compute evaluation metrics for NWM-USGS pairs.                                                                                       │
│ evaluate   Run standard evaluation including download, pair, and compute. Parameters set in configuration file.                                 │
│ export     Export evaluation metrics to CSV.                                                                                                    │
│ display    Launch graphical application in the browser. Shutdown the application using ctrl+c.                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
Note that each command (`download`, `pair`, `compute`, etc. will show additional information using `--help`)

### Standard Usage
Generally, users will want to run the `evaluate` and `display` commands in sequence to generate and explore NWM evaluations. The `evaluate` command will run generate standard evaluation metrics given the parameters in the JSON configuration file. For example, the configuration snippet below will run an evaluation labeled `test_run_1` on National Water Model forecasts and analyses from 2024-10-01 to 2024-10-03. Running `nwm-explorer evaluate` will `download` NWM and USGS time series, `pair` time series, and `compute` evaluation metrics. After `compute` completes, metrics can be explored using the GUI by running `nwm-explorer display`.
```json
# config.json
    "evaluations": [
        {
            "label": "test_run_1",
            "start_time": "2024-10-01",
            "end_time": "2024-10-03"
        }
    ],
```

## Graphical User Interface

The GUI includes many options for exploring evaluation results including mapping of metrics, filtering by lead time or confidence bounds, regional histograms, hydrographs, and site information.

![GUI](https://raw.githubusercontent.com/jarq6c/nwm-explorer/main/images/gui.png)
