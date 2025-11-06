# StyloFlora

## Setup

1. Download `XLCoST_data` from [here](https://drive.google.com/file/d/1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6/edit) and place it
in `${ROOT}/data/`.

> If you're working with PyCharm, it's recommended to exclude the `data` directory from indexing in
`File | Settings | Project: PROJECT_NAME | Project Structure` and from SonarLint analysis in
`File | Settings | Other Settings | SonarLint | File Exclusions` to avoid performance issues in the IDE.
2. Install the packaged framework by running `pip install -e .`.
3. Add StyleX JAR path to `CLASSPATH` environment variable; ensure PICT executable can be found in `PATH` environment variable.

## Structure

```
.
├── configs         # configurations
├── data            # benchmarks and datasets
├── logs            # logs while running
├── notebooks       # notebooks for exploration
├── README.md
├── results         # outputs for further analysis
├── scripts         # helper scripts
├── src             # source code of StyloFlora
└── targets         # compiled code while running
```

## Usage

Refer to scripts in `scripts/experiments`.
Example usage:
```bash
python3 -m scripts.experiments.run_evaluation \
  -d xCodeEval -m gemini-2.0-flash -t code_translation \
  --src-lang java --dst-lang python --result-dir results --log-path logs/StyloFlora.log \
  -n 100 --seed 42 -rv
```
