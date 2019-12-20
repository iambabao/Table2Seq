import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.wikientity import WikiEntity


class Evaluator:
    def __init__(self, key):
        self.key = key

    def get_bleu_score(self, truth_file, pred_file, to_lower):
        f_truth = open(truth_file, 'r', encoding='utf-8')
        f_pred = open(pred_file, 'r', encoding='utf-8')

        list_of_references = []
        hypotheses = []
        for truth in f_truth:

            truth = WikiEntity(truth)
            if len(truth.get_box()) == 0:
                continue
            truth = truth.get_desc()
            if to_lower:
                truth = truth.lower()
            truth = truth.split()

            pred = f_pred.readline()
            pred = json.loads(pred)
            pred = pred[self.key]
            if to_lower:
                pred = pred.lower()
            pred = pred.split()

            list_of_references.append([truth])
            hypotheses.append(pred)

        bleu1 = 100 * corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.), SmoothingFunction().method4)
        bleu2 = 100 * corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.), SmoothingFunction().method4)
        bleu3 = 100 * corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.), SmoothingFunction().method4)
        bleu4 = 100 * corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25), SmoothingFunction().method4)
        print('{:>.4f}, {:>.4f}, {:>.4f}, {:>.4f}'.format(bleu1, bleu2, bleu3, bleu4))
        return (bleu1 + bleu2 + bleu3 + bleu4) / 4
