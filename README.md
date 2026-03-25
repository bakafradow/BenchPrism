# BenchPrism

## Introduction

This is the official repository for the paper *The Stylistic Blind Spot: Uncovering the Hidden Implicit Bias of Coding Style on LLM Code Evaluation*, accepted by FSE-IVR '26.

BenchPrism is a framework to automatically disperse benchmarks towards diverse coding styles and evaluate LLMs on them.

## Structure

```
.
├── artifacts           # additional artifacts not fully presented in the paper
│   ├── RANKING.pdf     # full version of Figure 2 in the paper
│   └── STYLES.md       # complete list of supported coding styles
├── configs             # configurations
├── README.md           # this file
├── scripts             # scripts with program entries
│   ├── dev             # scripts assisting module development
│   ├── experiment      # scripts for research experiments
│   ├── postprocessing  # scripts normalizing model outputs, used after inference
│   └── preprocessing   # scripts fixing dataset issues, used before experiments
└── src                 # source code of BenchPrism module
```

## Setup

### Python Environment

Simply install the BenchPrism module by running:
```bash
pip install -e .
```

### Datasets

BenchPrism currently supports 7 benchmarks: xCodeEval, CodeScope, CodeMMLU, CRUXEval-X, ClassEval-T, CoderUJB and TestBench.

Some of them are pulled from HuggingFace Hub automatically, while others need to be downloaded manually with paths specified in configuration file (see below).

> [!tip]
> If you're working with PyCharm, it's recommended to exclude your data directories from indexing in `File | Settings | Project: PROJECT_NAME | Project Structure` and from SonarLint analysis in `File | Settings | Other Settings | SonarLint | File Exclusions` to avoid heavy analysis.

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

To evaluate model generated code, Windows environment is required due to some libraries used in ClassEval-T. The neatest way is to use WSL2 with MSYS2 configured on Windows. It's required to specify `MSYS2_ROOT` environment variable in dotenv file.

#### CoderUJB

Subset for code repair is automatically pulled from HuggingFace Hub; subset for defect detection requires manual construction with code from https://github.com/ZZR0/CoderUJB, setting `few_shot` to `-1` to get similar prompts to those in repair task.

To evaluate code repair tasks, `defects4j` environment is required. See README in CoderUJB repository for setup. Since it requires JDK 11 which conflicts with Styler's requirement (see below), it's recommended to specify `D4J_JAVA_HOME` environment variable in dotenv file.

#### TestBench

The dataset can be cloned from https://github.com/iSEngLab/TestBench; the `java_project` directory can be downloaded from the link in TestBench README and should be placed to `${testbench_root}/java_project`.

To evaluate model generated unit tests, `jacoco` tool and `pitest` tool are required. The JAR paths of `jacocoagent.jar`, `jacococli.jar` and `pitest.jar` should be added to `CLASSPATH` environment variable.

Due to Maven environment problems, especially those related to versions of dependencies such as `junit-jupiter-api` and `junit-jupiter-engine`, the `pom.xml` files in the repositories may need manual adjustment. JDK 17 is recommended for minimal adjustments.

### Style Transformer

#### Tool

A Java tool that performs style transfer across widely-investigated coding styles in an extract-and-apply manner is provided.
It is glued together with BenchPrism by the `transformer` module and the specifications in `configs/style_options.yaml`.

Add the path of JAR (available below) to `CLASSPATH` environment variable and ensure JDK version is at least 17.
When `--no-transform` flag is set while running an experiment, the transformer is unused and the configuration is not necessary.

#### PICT

PICT executable can be built from source at https://github.com/microsoft/pict. After building, add its path to `PATH` environment variable.
Also, experiments can be run without it when `--no-transform` flag is set.

### Configuration

Settings regarding datasets, transformation, inference and metrics can be found in `configs/settings.yaml`.

Base URL and API key for remote models should be specified in dotenv file. See `.env.example`.

## Usage

All scripts are recommended to be run in `python -m` manner staying in the root directory.
Use `--help` flag to see their available options.

## How to Reproduce

> [!important]
> Due to GitHub storage limits, the zipped experiment results in the paper and the JAR of style transformer are hosted at https://zenodo.org/records/18348861.

The rough steps to reproduce experiments are shown in `scripts/experiment/run_all.sh`, and the settings in `configs/settings.yaml` are aligned with the paper.

For convenience, transformation, inference and evaluation can be done separately for flexibility using `--no-transform`/`-T`, `--no-inference`/`-I` and `--no-evaluation`/`-E` flags.
Here's an example:
```bash
# only pick candidates and transform, producing a candidate cache file and a variant cache file
python3 -m scripts.experiment.run -d CodeScope -m openai:gpt-5-mini -t code_translation --src-lang java --dst-lang cpp --result-dir results --log-path logs/BenchPrism.log --seed 42 -IE
# only inference, requiring cached candidates and variants and producing a output cache file
python3 -m scripts.experiment.run -d CodeScope -m openai:gpt-5-mini -t code_translation --src-lang java --dst-lang cpp --result-dir results --log-path logs/BenchPrism.log --seed 42 -TE
# only evaluation, requiring cached candidates, variants, and outputs, producing a result file
python3 -m scripts.experiment.run -d CodeScope -m openai:gpt-5-mini -t code_translation --src-lang java --dst-lang cpp --result-dir results --log-path logs/BenchPrism.log --seed 42 -TI
```
All the cache files are stored in the directory specified by `--result-dir`, so it's recommended to use a fixed directory to utilize cache files.
In particular, if one wants to only reproduce the evaluation results without re-transforming and re-infering, place corresponding candidate file, variant file, and output file to the directory specified by `--result-dir` and make use of `--no-transform` and `--no-inference` flags.
