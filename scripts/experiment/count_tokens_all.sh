for dataset in xCodeEval CodeScope CodeMMLU CoderUJB CRUXEval-X TestBench ClassEval-T
do
  for model in gpt-5-mini gemini-2.5-flash qwen2.5-7b-instruct phi-4-mini-reasoning codegeex4-all-9b codellama-7b
  do
    python3 -m scripts.count_tokens -d "$dataset" -m "$model" --src-lang java
  done
done
