"""Recompute binary metrics from a saved evaluation without loading a model."""

import argparse
import json
import math
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def metrics(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    tp = sum(score > 0.5 for score in positives)
    fp = sum(score > 0.5 for score in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    winning_pairs = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
    return {
        'accuracy': ratio(tp + tn, len(labels)),
        'precision': precision,
        'recall': recall,
        'f1': ratio(2 * precision * recall, precision + recall),
        'auc': ratio(winning_pairs, len(positives) * len(negatives)),
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def audit(path):
    report = json.loads(path.read_text())
    labels = report['labels']
    score_sets = report['scores']
    assert set(labels) == {0, 1} and len(set(report['paths'])) == len(labels), path
    individual = [values for key, values in score_sets.items() if key != 'ensemble']
    assert len(individual) == 4
    mean_scores = [sum(values) / len(values) for values in zip(*individual)]
    for actual, expected in zip(score_sets['ensemble'], mean_scores):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8), path

    print(f'{path.name}: {labels.count(0)} authentic, {labels.count(1)} manipulated')
    for key, scores in score_sets.items():
        assert len(scores) == len(labels) and all(0 <= s <= 1 for s in scores), key
        actual = metrics(labels, scores)
        expected = report['metrics'][key]
        assert actual['confusion_matrix'] == expected['confusion_matrix'], key
        for field in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
            assert math.isclose(actual[field], expected[field], rel_tol=0, abs_tol=1e-8), (key, field)
        print(f'  {key:18} accuracy={actual["accuracy"]:.4f} auc={actual["auc"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    for result in parser.parse_args().results:
        audit(result)
