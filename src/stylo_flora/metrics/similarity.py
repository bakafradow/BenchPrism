from collections.abc import Callable
from collections.abc import Sequence as Seq

import codebleu as cb
import evaluate
import numpy as np
from nltk.translate.meteor_score import meteor_score

from ..logger import logger

logger.info('Initializing metrics from evaluate library...')
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')


def ensure_equal_lengths(func: Callable) -> Callable:
  def wrapper(prd: Seq[str], ref: Seq[str], *args, **kwargs):
    if len(prd) != len(ref):
      raise ValueError(f'Predictions and References must have the same length, got {len(prd)} and {len(ref)}.')
    return func(prd, ref, *args, **kwargs)
  return wrapper


@ensure_equal_lengths
def calc_codebleu(prd: Seq[str], ref: Seq[str], lang: str) -> dict[str, float]:
  lang_to_name = {l: l for l in cb.AVAILABLE_LANGS} | {'cs': 'c_sharp', 'js': 'javascript'}
  return cb.calc_codebleu(references=ref, predictions=prd,
                          lang=lang_to_name[lang], weights=(.25, .25, .25, .25), tokenizer=None)


@ensure_equal_lengths
def calc_bleu(prd: Seq[str], ref: Seq[str]) -> float:
  return bleu.compute(predictions=prd, references=[[sentence] for sentence in ref])['bleu']


@ensure_equal_lengths
def calc_rouge(prd: Seq[str], ref: Seq[str]) -> dict:
  return rouge.compute(predictions=prd, references=[[sentence] for sentence in ref])


@ensure_equal_lengths
def calc_meteor(prd: Seq[str], ref: Seq[str]) -> float:
  scores = [meteor_score(references=[sentence_ref.split()], hypothesis=sentence_prd.split())
            for sentence_prd, sentence_ref in zip(prd, ref)]
  return np.mean(scores)


@ensure_equal_lengths
def calc_bertscore(prd: Seq[str], ref: Seq[str]) -> dict:
  return bertscore.compute(predictions=prd, references=ref, lang='en')
