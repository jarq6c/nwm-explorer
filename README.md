# National Water Model Evaluation Explorer

A web-based application used to explore National Water Model output and evaluation metrics.

## Installation
```bash
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install -U pip wheel
(env) $ pip install nwm_explorer
```

## Usage

```console
(env) $ nwm-explorer --help
Usage: nwm-explorer [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  evaluate  Run standard evaluation and generate parquet files.
  export    Export NWM evaluation data to CSV format.
  metrics   Export NWM evaluation metrics to CSV format.
```

```console
(env) $ nwm-explorer evaluate --help
Usage: nwm-explorer evaluate [OPTIONS]

  Run standard evaluation and generate parquet files.

  Example:

  nwm-explorer evaluate -s 20231001 -e 20240101

Options:
  -s, --startDT TIMESTAMP  Start datetime  [required]
  -e, --endDT TIMESTAMP    End datetime  [required]
  -d, --directory PATH     Data directory (./data)
  --help                   Show this message and exit.
```

```console
(env) $ nwm-explorer export --help
Usage: nwm-explorer export [OPTIONS] {alaska|conus|hawaii|puertorico} {analysi
                           s_assim_extend_alaska_no_da|analysis_assim_extend_n
                           o_da|analysis_assim_hawaii_no_da|analysis_assim_pue
                           rtorico_no_da|medium_range_mem1|medium_range_blend|
                           medium_range_no_da|usgs}

  Export NWM evaluation data to CSV format.

  Example:

  nwm-explorer export alaska analysis_assim_extend_alaska_no_da -s 20231001 -e
  20240101 -o alaska_analysis_data.csv

Options:
  -o, --output FILENAME       Output file path
  -s, --startDT TIMESTAMP     Start datetime  [required]
  -e, --endDT TIMESTAMP       End datetime  [required]
  --comments / --no-comments  Enable/disable comments in output, enabled by
                              default
  --header / --no-header      Enable/disable header in output, enabled by
                              default
  -d, --directory PATH        Data directory (./data)
  --help                      Show this message and exit.
```

```console
(env) $ nwm-explorer metrics --help
Usage: nwm-explorer metrics [OPTIONS] {alaska|conus|hawaii|puertorico} {analys
                            is_assim_extend_alaska_no_da|analysis_assim_extend
                            _no_da|analysis_assim_hawaii_no_da|analysis_assim_
                            puertorico_no_da|medium_range_mem1|medium_range_bl
                            end|medium_range_no_da}

  Export NWM evaluation metrics to CSV format.

  Example:

  nwm-explorer metrics alaska analysis_assim_extend_alaska_no_da -s 20231001
  -e 20240101 -o alaska_analysis_metrics.csv

Options:
  -o, --output FILENAME       Output file path
  -s, --startDT TIMESTAMP     Start datetime  [required]
  -e, --endDT TIMESTAMP       End datetime  [required]
  --comments / --no-comments  Enable/disable comments in output, enabled by
                              default
  --header / --no-header      Enable/disable header in output, enabled by
                              default
  -d, --directory PATH        Data directory (./data)
  --help                      Show this message and exit.
```
