from collections.abc import Sequence


def calc_macro_f1(pred: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]) -> float:
  """
  Calculate the macro F1 score for the predicted and gold tags.
  :param pred: the predicted tags
  :param gold: the gold standard tags
  :return: the macro F1 score
  """
  if len(pred) != len(gold):
      raise ValueError('The size of 2 snippet sequences should equal.')

  precision = {}
  recall = {}
  f1_scores = {}
  for tag in set.union(*(set(tags) for tags in pred), *(set(tags) for tags in gold)):
      tp = sum(1 for p, g in zip(pred, gold) if tag in p and tag in g)
      fp = sum(1 for p in pred if tag in p) - tp
      fn = sum(1 for g in gold if tag in g) - tp

      precision[tag] = tp / (tp + fp) if tp + fp else .0
      recall[tag] = tp / (tp + fn) if tp + fn else .0
      f1_scores[tag] = 2 * (precision[tag] * recall[tag]) / (precision[tag] + recall[tag]) \
                       if precision[tag] + recall[tag] else .0

  return sum(f1_scores.values()) / len(f1_scores) if f1_scores else .0
