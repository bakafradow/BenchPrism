# Code Translator

## Setup

Download `XLCoST_data` from [here](https://drive.google.com/file/d/1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6/edit) and place it
in `${ROOT}/data/`.

> If you're working with PyCharm, it's recommended to exclude the `data` directory from indexing in
`File | Settings | Project: PROJECT_NAME | Project Structure` and from SonarLint analysis in
`File | Settings | Other Settings | SonarLint | File Exclusions`, to avoid performance issues in the IDE.

Install the required packages by running `pip install -r requirements.txt`.

## Usage

See `python3 main.py --help`.

## TODOs

- [x] How does the transformer work?
- [x] How to evaluate the code set with CodeBLEU, etc.?
- [ ] Pick up datasets for testing
  - humanEvalX
    1. load source code from HF
    2. mutate code (optional)
    3. translate code by model
    4. generate JSON
    5. run evaluate_humaneval_x.sh (in Docker)
  - xCodeEval
    1. load source code from HF
    2. mutate code (optional)
    3. translate code by model
    4. HOW DOES IT WORK?
  - XLCoST?
    - only evaluator with CodeBLEU
  - CodeXGLUE?
    - only evaluator with CodeBLEU
  - G-TransEval?
    - too small
