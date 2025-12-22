# StyloFlora

## Introduction

StyloFlora is a framework to augment and evaluate benchmarks for various LLM code tasks.

## Structure

```
.
├── configs             # configurations
├── README.md
├── scripts             # scripts with program entries
│   ├── dev             # scripts assisting module development
│   ├── experiment      # scripts for research experiments
│   ├── postprocessing  # scripts normalizing model outputs, used after inference
│   ├── preprocessing   # scripts fixing dataset issues, used before experiments
└── src                 # source code of StyloFlora module
```

## Setup

### Python Environment

Simply install the StyloFlora module by running:
```bash
pip install -e .
```

### Datasets

StyloFlora currently supports 7 benchmarks: xCodeEval, CodeScope, CodeMMLU, CRUXEval-X, ClassEval-T, CoderUJB and TestBench.

Some of them are pulled from HuggingFace Hub automatically, while others need to be downloaded manually with paths specified in configuration file (see below).

> If you're working with PyCharm, it's recommended to exclude your data directories from indexing in
`File | Settings | Project: PROJECT_NAME | Project Structure` and from SonarLint analysis in
`File | Settings | Other Settings | SonarLint | File Exclusions` to avoid heavy analysis.

#### xCodeEval

The dataset can be cloned from https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval.

#### CodeScope

The dataset can be cloned from https://github.com/WeixiangYAN/CodeScope.

To evaluate test generation tasks, `jacoco` tool is required. The JAR path of `jacocoagent.jar` and `jacococli.jar` should be added to `CLASSPATH` environment variable.

#### CodeMMLU

Automatically pulled from HuggingFace Hub.

#### CRUXEval-X

Automatically pulled from HuggingFace Hub.

#### ClassEval-T

The dataset can be cloned from https://github.com/wLinHoo/ClassEval-T.

To evaluate model generated code, Windows environment is required due to some libraries used in ClassEval-T. The neatest way is to use WSL2 with MSYS2 configured on Windows. It's recommended to specify `MSYS2_ROOT` environment variable in dotenv file.

#### CoderUJB

Subset for code repair is automatically pulled from HuggingFace Hub; subset for defect detection requires manual construction with code from https://github.com/ZZR0/CoderUJB, setting `few_shot` to `-1` to get similar prompts to those in repair task.

To evaluate code repair tasks, `defects4j` environment is required. See README in CoderUJB repository for setup. Since it requires JDK 11 which conflicts with Styler's requirement (see below), it's recommended to specify `D4J_JAVA_HOME` environment variable in dotenv file.

#### TestBench

The dataset can be cloned from https://github.com/iSEngLab/TestBench with Java projects downloaded from the link in the README.

To evaluate model generated unit tests, `jacoco` tool and `pitest` tool are required. The JAR paths of `jacocoagent.jar`, `jacococli.jar` and `pitest.jar` should be added to `CLASSPATH` environment variable.

### Styler

#### StyleX

Add StyleX JAR path to `CLASSPATH` environment variable and ensure JDK version is at least 17.

#### PICT

PICT executable can be built from source at https://github.com/microsoft/pict. After building, add its path to `PATH` environment variable.

### Configuration

Settings regarding datasets, transformation, inference and metrics can be found in `configs/settings.yaml`.

Base URL and API key for remote models should be specified in dotenv file. See `.env.example`.

## Usage

All scripts are recommended to be run in `python -m` manner staying in the root directory.

Specifically, for the experiment script `scripts/experiment/run.py`, transformer

Example usage:
```bash
python -m scripts.experiment.run \
  -d xCodeEval -m gemini-2.5-flash -t code_translation \
  --src-lang java --dst-lang python --result-dir results --log-path logs/StyloFlora.log \
  --seed 42 -rvTI -i :1000
```
