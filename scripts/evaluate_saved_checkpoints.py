"""Evaluate saved detectors with validation-only calibration and held-out metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from data_loader.audio_processor import AudioProcessor
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

from data_loader.datasets import DeepfakeDataset, load_split_manifest
from models.baselines import LipSyncDetector, MesoNet, XceptionNet
from models.multimodal_model import MultimodalModel

MODEL_NAMES = ('multimodal', 'mesonet', 'xceptionnet', 'lipsync')


def pick_threshold(labels, scores):
    """Maximize validation F1, using accuracy to break ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = (scores > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, prediction, average='binary', zero_division=0
        )
        accuracy = accuracy_score(labels, prediction)
        key = (float(f1), float(accuracy))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def summarize(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average='binary', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(labels, prediction)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float('nan'),
        'confusion_matrix': confusion_matrix(labels, prediction).tolist(),
    }


def score_loader(models, loader, paths, silent_audio):
    scores = {name: [] for name in models}
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            video = batch['video'].cuda()
            audio = batch['audio'].cuda()
            if silent_audio:
                audio = torch.zeros_like(audio)
            if video.shape[1] != 30 or audio.shape[-1] < 8:
                raise ValueError(f'Incomplete video decode: {paths[index]}')
            for name, model in models.items():
                output = (model.process_video(video, audio)
                          if name in ('multimodal', 'lipsync')
                          else model.process_video(video))
                output = output[0] if isinstance(output, tuple) else output
                if getattr(model, 'output_type', 'sigmoid') == 'logits':
                    output = torch.sigmoid(output)
                scores[name].append(output.item())
    result = {name: np.asarray(values, dtype=float) for name, values in scores.items()}
    for name, values in result.items():
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite prediction: {name}')
    return result


def calibrated_ensemble(validation_metrics):
    if not validation_metrics:
        weights = np.ones(len(MODEL_NAMES), dtype=float)
    else:
        weights = np.asarray([
            max(validation_metrics[name]['auc'] - 0.5, 0.0)
            if np.isfinite(validation_metrics[name]['auc']) else 0.0
            for name in MODEL_NAMES
        ], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(MODEL_NAMES), dtype=float)
    weights /= weights.sum()
    uniform = np.ones(len(MODEL_NAMES), dtype=float) / len(MODEL_NAMES)
    weights = 0.5 * weights + 0.5 * uniform
    return {name: float(weight) for name, weight in zip(MODEL_NAMES, weights)}


def combine_scores(scores, weights):
    return sum(scores[name] * weights[name] for name in MODEL_NAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights-dir', type=Path, default=Path('weights'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--split-manifest', type=Path)
    selection.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-name', default='metrics.json')
    parser.add_argument('--require-audio', action='store_true')
    parser.add_argument('--silent-audio', action='store_true')
    parser.add_argument('--skip-face-crop', action='store_true')
    args = parser.parse_args()
    if args.require_audio and args.silent_audio:
        parser.error('--require-audio and --silent-audio are mutually exclusive')
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.json'):
        parser.error('--output-name must be a JSON filename')
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)

    constructors = {
        'multimodal': lambda: MultimodalModel(pretrained=False),
        'mesonet': MesoNet,
        'xceptionnet': lambda: XceptionNet(pretrained=False),
        'lipsync': LipSyncDetector,
    }
    models = {}
    for name, constructor in constructors.items():
        path = args.weights_dir / f'trained_model_{name}_best.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = constructor()
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
        model.output_type = checkpoint.get('output_type', 'sigmoid')
        models[name] = model.cuda().eval()
        print(f'{path}: loaded {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    splits = load_split_manifest(args.split_manifest) if args.split_manifest else None
    if args.split_manifest:
        datasets = {Path(args.output_name).stem: None}
    elif args.data_dir:
        datasets = {args.data_dir.name: args.data_dir}
    else:
        datasets = {name: Path('data') / name for name in ('wilddeepfake', 'dfdc', 'ff_plus/c23', 'ff_plus/c40')}

    dataset_kwargs = dict(require_audio=args.require_audio,
                          skip_face_crop=args.skip_face_crop)
    for dataset, directory in datasets.items():
        if splits:
            paths, labels = map(list, zip(*splits['test']))
            seen = {Path(path).resolve() for key in ('train', 'validation') for path, _ in splits[key]}
            if any(Path(path).resolve() in seen for path in paths) or set(labels) != {0, 1}:
                raise ValueError('Invalid held-out test split')
            val_pairs = splits.get('validation', [])
            val_paths, val_labels = map(list, zip(*val_pairs)) if val_pairs else ([], [])
        else:
            paths, labels = [], []
            for label, category in enumerate(('real', 'fake')):
                clips = sorted((directory / category).glob('*.mp4'))
                if not clips:
                    raise ValueError(f'No {category} clips in {dataset}')
                paths.extend(map(str, clips))
                labels.extend([label] * len(clips))
            val_paths, val_labels = [], []

        if args.require_audio:
            audio_processor = AudioProcessor()
            for path in paths:
                waveform = audio_processor.extract_audio(path)
                if not np.isfinite(waveform).all() or float(np.max(np.abs(waveform))) < 1e-4:
                    raise ValueError(f'No usable audio samples: {path}')
            print(f'Decoded non-silent audio from all {len(paths)} clips', flush=True)

        print(dataset, dict(Counter(labels)), flush=True)
        loader = DataLoader(DeepfakeDataset(paths, labels, **dataset_kwargs),
                            batch_size=1, num_workers=4)
        test_scores = score_loader(models, loader, paths, args.silent_audio)

        validation_metrics = {}
        validation_scores = {}
        thresholds = {name: 0.5 for name in MODEL_NAMES}
        if val_paths and len(set(val_labels)) == 2:
            val_loader = DataLoader(DeepfakeDataset(val_paths, val_labels, **dataset_kwargs),
                                    batch_size=1, num_workers=4)
            validation_scores = score_loader(models, val_loader, val_paths, args.silent_audio)
            thresholds = {name: pick_threshold(val_labels, validation_scores[name])
                          for name in MODEL_NAMES}
            validation_metrics = {name: summarize(val_labels, validation_scores[name], thresholds[name])
                                  for name in MODEL_NAMES}

        weights = calibrated_ensemble(validation_metrics)
        if validation_scores:
            val_ensemble = combine_scores(validation_scores, weights)
            thresholds['ensemble'] = pick_threshold(val_labels, val_ensemble)
            validation_metrics['ensemble'] = summarize(
                val_labels, val_ensemble, thresholds['ensemble']
            )
        else:
            thresholds['ensemble'] = 0.5
        test_scores['ensemble'] = combine_scores(test_scores, weights)

        metrics = {name: summarize(labels, values, thresholds[name])
                   for name, values in test_scores.items()}
        payload = {
            'paths': paths,
            'labels': labels,
            'scores': {name: values.tolist() for name, values in test_scores.items()},
            'metrics': metrics,
            'thresholds': thresholds,
            'ensemble_weights': weights,
            'validation_metrics': validation_metrics,
            'validation_labels': val_labels,
            'validation_scores': {name: values.tolist() for name, values in validation_scores.items()},
        }
        output_path = args.output / args.output_name
        output_path.write_text(json.dumps(payload, indent=2))
        print(dataset, json.dumps({'metrics': metrics, 'thresholds': thresholds,
                                   'ensemble_weights': weights}), flush=True)


if __name__ == '__main__':
    main()
