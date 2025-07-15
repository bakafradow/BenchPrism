# StyloFlora

## Setup

1. Download `XLCoST_data` from [here](https://drive.google.com/file/d/1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6/edit) and place it
in `${ROOT}/data/`.

> If you're working with PyCharm, it's recommended to exclude the `data` directory from indexing in
`File | Settings | Project: PROJECT_NAME | Project Structure` and from SonarLint analysis in
`File | Settings | Other Settings | SonarLint | File Exclusions` to avoid performance issues in the IDE.
2. Install the packaged framework by running `pip install -e .`.
3. Package `code-style-transformer` with Maven, configure the path to the JAR file and your JVM in `settings.yml`.

## Structure

```
.
├── data            # benchmarks and datasets
├── experiments     # scripts for experiments
├── logs            # logs while running
├── notebooks       # notebooks for exploration
├── README.md
├── results         # outputs for further analysis
├── scripts         # helper scripts
├── settings.yml    # configuration file
├── src             # source code of StyloFlora
└── targets         # compiled code while running
```

## Usage

Refer to scripts in `experiments`.
Example usage:
```bash
python3 experiments/evaluate_translation.py -d HumanEvalX -m gemini-2.0-flash --src-lang java --dst-lang python -n 100 --seed 42
```
