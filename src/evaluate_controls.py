"""Evaluate frame-order and audio-silence controls for a trained detector."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader

from data_loader.datasets import DeepfakeDataset, load_split_manifest
from models.multimodal_model import MultimodalModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path):
    raw = json.loads(Path(path).read_text())
    splits = load_split_manifest(path)
    categories = {}
    for records in raw.get('splits', {}).values():
        for record in records:
            if not isinstance(record, dict):
                continue
            candidate = Path(record['path'])
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            categories[str(candidate.resolve())] = record.get('category', 'unknown')
    return splits, categories


def scores_for(model, loader, condition, device, output_type):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            if condition in ('frame_shuffled', 'both'):
                permutation = torch.randperm(video.shape[1], device=device)
                video = video[:, permutation]
            if condition in ('audio_silent', 'both'):
                audio = torch.zeros_like(audio)
            logits = model(video, audio)
            scores = torch.sigmoid(logits) if output_type == 'logits' else logits
            values.extend(scores.squeeze(-1).cpu().numpy().tolist())
    return np.asarray(values, dtype=float)


def pick_threshold(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    useful = thresholds[1:]
    if len(useful) == 0:
        return 0.5
    return float(useful[np.argmax(tpr[1:] - fpr[1:])])


def summarize(labels, scores, threshold):
    labels = np.asarray(labels)
    predictions = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    result = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
    }
    result['auc'] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float('nan')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--require-audio', action='store_true')
    parser.add_argument('--skip-face-crop', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device('cuda')

    splits, categories = load_manifest(args.split_manifest)
    dataset_kwargs = dict(require_audio=args.require_audio,
                          skip_face_crop=args.skip_face_crop)

    def make_loader(split):
        paths, labels = map(list, zip(*splits[split]))
        dataset = DeepfakeDataset(paths, labels, augment=False, **dataset_kwargs)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4), paths, labels

    val_loader, _, val_labels = make_loader('validation')
    test_loader, test_paths, test_labels = make_loader('test')

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MultimodalModel(pretrained=False).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    output_type = checkpoint.get('output_type', 'sigmoid')

    val_scores = scores_for(model, val_loader, 'original', device, output_type)
    threshold = pick_threshold(val_labels, val_scores)

    conditions = ('original', 'frame_shuffled', 'audio_silent', 'both')
    results = {}
    for condition in conditions:
        scores = scores_for(model, test_loader, condition, device, output_type)
        results[condition] = summarize(test_labels, scores, threshold)
        results[condition]['scores'] = scores.tolist()

    original_scores = np.asarray(results['original']['scores'])
    category_metrics = {}
    for category in sorted(set(categories.get(path, 'unknown') for path in test_paths)):
        indices = [i for i, path in enumerate(test_paths)
                   if categories.get(path, 'unknown') == category]
        if not indices:
            continue
        subset = summarize(np.asarray(test_labels)[indices], original_scores[indices], threshold)
        subset['count'] = len(indices)
        category_metrics[category] = subset

    payload = {
        'checkpoint': str(args.checkpoint),
        'split_manifest': str(args.split_manifest),
        'output_type': output_type,
        'threshold': threshold,
        'validation_scores': val_scores.tolist(),
        'validation_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels,
        'conditions': results,
        'category_metrics': category_metrics,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'scores'}
                      for name, result in results.items()}, indent=2))
    print(f'Controls saved to {args.output}')


if __name__ == '__main__':
    main()
