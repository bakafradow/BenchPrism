#!/bin/bash

set -e

benchmarks=(
    xCodeEval
    CodeScope
    CodeMMLU
    CRUXEval-X
    ClassEval-T
    CoderUJB
    TestBench
)

models=(
    gpt-5-mini
    gemini-2.5-flash
    microsoft/phi-4
    Qwen/Qwen2.5-14B-Instruct
    codegeex-4
    codellama/codellama-7b-instruct-hf
)

for dataset in "${benchmarks[@]}"; do
  for model in "${models[@]}"; do
    python3 -m scripts.count_tokens -d "$dataset" -m "$model" --src-lang java
  done
done
