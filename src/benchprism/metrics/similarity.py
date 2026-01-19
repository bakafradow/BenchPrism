from collections.abc import Callable
from collections.abc import Sequence as Seq
from functools import cache

import codebleu as cb
import evaluate
import numpy as np
from nltk.translate.meteor_score import meteor_score

from ..logger import logger


def ensure_equal_lengths(func: Callable) -> Callable:
  def wrapper(prd: Seq[str], ref: Seq[str], *args, **kwargs):
    if len(prd) != len(ref):
      raise ValueError(f'Predictions and References must have the same length, got {len(prd)} and {len(ref)}.')
    return func(prd, ref, *args, **kwargs)
  return wrapper


@ensure_equal_lengths
def calc_codebleu(prd: Seq[str], ref: Seq[str], lang: str) -> dict[str, float]:
  lang_to_name = {l: l for l in cb.AVAILABLE_LANGS} | {'cs': 'c_sharp', 'js': 'javascript'}
  return cb.calc_codebleu(references=list(ref), predictions=list(prd),
                          lang=lang_to_name[lang], weights=(.25, .25, .25, .25), tokenizer=None)


@cache
def _get_bleu():
  logger.info('Initializing BLEU from evaluate library...')
  return evaluate.load('bleu')


@ensure_equal_lengths
def calc_bleu(prd: Seq[str], ref: Seq[str]) -> float:
  bleu = _get_bleu()
  return bleu.compute(predictions=prd, references=[[sentence] for sentence in ref])['bleu']


@cache
def _get_rouge():
  logger.info('Initializing ROUGE from evaluate library...')
  return evaluate.load('rouge')


@ensure_equal_lengths
def calc_rouge(prd: Seq[str], ref: Seq[str]) -> dict:
  rouge = _get_rouge()
  return rouge.compute(predictions=prd, references=[[sentence] for sentence in ref])


@ensure_equal_lengths
def calc_meteor(prd: Seq[str], ref: Seq[str]) -> float:
  scores = [meteor_score(references=[sentence_ref.split()], hypothesis=sentence_prd.split())
            for sentence_prd, sentence_ref in zip(prd, ref)]
  return np.mean(scores)


@cache
def _get_bertscore():
  logger.info('Initializing BERTScore from evaluate library...')
  return evaluate.load('bertscore')


@ensure_equal_lengths
def calc_bertscore(prd: Seq[str], ref: Seq[str]) -> dict:
  bertscore = _get_bertscore()
  return bertscore.compute(predictions=prd, references=ref, lang='en')
