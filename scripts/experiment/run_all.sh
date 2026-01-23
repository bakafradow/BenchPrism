#!/bin/bash

set -e

# Layout in data_dir:
# data
# ├── ClassEval-T
# ├── CoderUJB
# ├── CodeScope
# ├── TestBench
# └── xCodeEval

data_dir=data
result_dir=results
log_path=logs/BenchPrism.log
seed=42
assistant_model=openai:THUDM/GLM-4-32B-0414

models=(
    gpt-5-mini
    gemini-2.5-flash
    phi-4
    Qwen2.5-14B-Instruct
    codegeex-4
    codellama-7b-instruct-hf
)

declare -A model_to_prefix
model_to_prefix['gpt-5-mini']='openai:'
model_to_prefix['gemini-2.5-flash']='gemini:'
model_to_prefix['phi-4']='openai:microsoft/'
model_to_prefix['Qwen2.5-14B-Instruct']='openai:Qwen/'
model_to_prefix['codegeex-4']='zhipu:'
model_to_prefix['codellama-7b-instruct-hf']='codellama/'

# --- xCodeEval ---
# Clone repo structure, then pull heavy LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval $data_dir/xCodeEval
cd $data_dir/xCodeEval && git lfs pull && cd - # TODO: only download problem_descriptions.jsonl unittest_db.json code_translation/test/Java.jsonl apr/test/Java.jsonl tag_classification/test/Java.jsonl

benchmark=xCodeEval
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang cpp --result-dir $result_dir --log-path $log_path --seed $seed -r -i :463
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang python --result-dir $result_dir --log-path $log_path --seed $seed -r -i :463
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_repair --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed -r -i :508
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code2tag --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed -r -i :1000

  # only support LLM-assisted amending
  python3 -m scripts.postprocessing.amend_outputs_xcodeeval_code2tag -f "$result_dir/outputs_xcodeeval_code2tag_java_with_${model}_seed$seed.jsonl" -m $assistant_model  
done

# --- CodeScope ---
git clone https://github.com/WeixiangYAN/CodeScope.git $data_dir/CodeScope

benchmark=CodeScope
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang cpp --result-dir $result_dir --log-path $log_path --seed $seed
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang python --result-dir $result_dir --log-path $log_path --seed $seed
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_repair --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed
  python3 -m scripts.experiment.run -d $benchmark -m $m -t test_generation --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed

  python3 -m scripts.postprocessing.amend_outputs_codescope_test_generation -f "$result_dir/outputs_codescope_test_generation_java_with_${model}_seed$seed.jsonl"
done

# --- CodeMMLU ---
benchmark=CodeMMLU
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t mcq_answering --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed -r -i :655

  python3 -m scripts.postprocessing.amend_outputs_codemmlu -f "$result_dir/outputs_codemmlu_mcq_answering_java_with_${model}_seed$seed.jsonl"
done

# --- CRUXEval-X ---
benchmark=CRUXEval-X
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t input_reasoning --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed
  python3 -m scripts.experiment.run -d $benchmark -m $m -t output_reasoning --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed

  python3 -m scripts.postprocessing.amend_outputs_cruxeval_x -f "$result_dir/outputs_cruxeval-x_input_reasoning_java_with_${model}_seed$seed.jsonl" -t input_reasoning
  python3 -m scripts.postprocessing.amend_outputs_cruxeval_x -f "$result_dir/outputs_cruxeval-x_output_reasoning_java_with_${model}_seed$seed.jsonl" -t output_reasoning
done

# --- ClassEval-T ---
git clone https://github.com/wLinHoo/ClassEval-T.git $data_dir/ClassEval-T
python3 -m scripts.preprocessing.amend_classeval_t $data_dir/ClassEval-T/ClassEval_T

benchmark=ClassEval-T
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang cpp --result-dir $result_dir --log-path $log_path --seed $seed

  python3 -m scripts.postprocessing.amend_classeval_t -f "$result_dir/outputs_classeval-t_code_translation_java_to_cpp_with_${model}_seed$seed.jsonl" --src-lang java --dst-lang cpp
done

# Clean up the cache since candidates for C++ and Python are not identical
rm "$result_dir/candidates_classeval-t_code_translation_java_seed$seed.json"

for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_translation --src-lang java --dst-lang python --result-dir $result_dir --log-path $log_path --seed $seed

  python3 -m scripts.postprocessing.amend_classeval_t -f "$result_dir/outputs_classeval-t_code_translation_java_to_python_with_${model}_seed$seed.jsonl" --src-lang java --dst-lang python
done

# --- CoderUJB ---
git clone https://github.com/ZZR0/CoderUJB.git $data_dir/CoderUJB
if [ ! -f "$data_dir/CoderUJB/datasets/data/task_defectdetection_bench_1111|2048.json" ]; then
  echo "Dataset for defect detection of CoderUJB not found."
  exit 1
fi

benchmark=CoderUJB
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t code_repair --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed
  python3 -m scripts.experiment.run -d $benchmark -m $m -t defect_detection --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed

  python3 -m scripts.postprocessing.amend_outputs_coderujb_code_repair -f "$result_dir/outputs_coderujb_code_repair_java_with_${model}_seed$seed.jsonl"
  python3 -m scripts.postprocessing.amend_outputs_coderujb_defect_detection -f "$result_dir/outputs_coderujb_defect_detection_java_with_${model}_seed$seed.jsonl"
done

# --- TestBench ---
git clone https://github.com/iSEngLab/TestBench.git $data_dir/TestBench

# Download heavy java_project.zip from Google Drive using gdown
pip install gdown -q
gdown 1syRdGJfvM7ZWlvEwuEBFrYDNrtw-NGzh -O $data_dir/TestBench/java_project.zip
unzip -q $data_dir/TestBench/java_project.zip -d $data_dir/TestBench/java_project  # TODO: since the zip contains a directory named `java_project`, do not create nested one
rm $data_dir/TestBench/java_project.zip # cleanup

python3 -m scripts.preprocessing.amend_testbench $data_dir/TestBench

benchmark=TestBench
for model in "${models[@]}"; do
  m="${model_to_prefix[$model]}$model"
  python3 -m scripts.experiment.run -d $benchmark -m $m -t test_generation --src-lang java --result-dir $result_dir --log-path $log_path --seed $seed
done
