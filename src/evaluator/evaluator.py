import json

from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
from src.wikientity import WikiEntity


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        refs = []
        with open(ref_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                ref = WikiEntity(line)
                if len(ref.get_box()) == 0:
                    continue
                ref = ref.get_desc()
                if to_lower:
                    ref = ref.lower()
                refs.append(ref)
        refs = {idx: [ref] for idx, ref in enumerate(refs)}

        hyps = []
        with open(hyp_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                hyp = json.loads(line)[self.key]
                if to_lower:
                    hyp = hyp.lower()
                hyps.append(hyp)
        hyps = {idx: [hyp] for idx, hyp in enumerate(hyps)}

        assert len(refs) == len(hyps)

        res = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    res[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                res[method] = score

        return res
