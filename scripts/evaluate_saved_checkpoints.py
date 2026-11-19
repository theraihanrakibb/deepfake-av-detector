"""Run the four saved detectors on the bundled clips and export per-video scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from moviepy import VideoFileClip
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src.data_loader.datasets import DeepfakeDataset
from src.models.baselines import LipSyncDetector, MesoNet, XceptionNet
from src.models.multimodal_model import MultimodalModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights-dir', type=Path, default=Path('weights'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--split-manifest', type=Path)
    selection.add_argument('--data-dir', type=Path)
    parser.add_argument('--require-audio', action='store_true')
    args = parser.parse_args()
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
        models[name] = model.cuda().eval()
        print(f'{path}: loaded {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    if args.split_manifest:
        datasets = {'held_out_test': None}
    elif args.data_dir:
        datasets = {args.data_dir.name: args.data_dir}
    else:
        datasets = {name: Path('data') / name for name in ('wilddeepfake', 'dfdc', 'ff_plus/c23', 'ff_plus/c40')}
    for dataset, directory in datasets.items():
        paths, labels = [], []
        if args.split_manifest:
            splits = json.loads(args.split_manifest.read_text())
            paths, labels = map(list, zip(*splits['test']))
            seen = {Path(path).resolve() for key in ('train', 'validation') for path, _ in splits[key]}
            if any(Path(path).resolve() in seen for path in paths) or set(labels) != {0, 1}:
                raise ValueError('Invalid held-out test split')
        else:
            for label, category in enumerate(('real', 'fake')):
                clips = sorted((directory / category).glob('*.mp4'))
                if not clips:
                    raise ValueError(f'No {category} clips in {dataset}')
                paths.extend(map(str, clips))
                labels.extend([label] * len(clips))
        if args.require_audio:
            for path in paths:
                with VideoFileClip(path) as clip:
                    if clip.audio is None:
                        raise ValueError(f'Missing audio track: {path}')
                    waveform = clip.audio.subclipped(0, min(clip.audio.duration, 5)).to_soundarray(fps=16000)
                    if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) == 0:
                        raise ValueError(f'No usable audio samples: {path}')
            print(f'Decoded non-silent audio from all {len(paths)} clips', flush=True)
        loader = DataLoader(DeepfakeDataset(paths, labels), batch_size=1, num_workers=4)
        scores = {name: [] for name in models}
        print(dataset, dict(Counter(labels)), flush=True)
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                video, audio = batch['video'].cuda(), batch['audio'].cuda()
                if video.shape[1] != 30 or audio.shape[-1] < 8:
                    raise ValueError(f'Incomplete video decode: {paths[index]}')
                for name, model in models.items():
                    output = model.process_video(video, audio) if name in ('multimodal', 'lipsync') else model.process_video(video)
                    output = output[0] if isinstance(output, tuple) else output
                    scores[name].append(output.item())
        scores['ensemble'] = np.mean(list(scores.values()), axis=0).tolist()
        metrics = {}
        for name, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite prediction: {dataset} / {name}')
            prediction = np.asarray(values) > 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average='binary', zero_division=0)
            metrics[name] = dict(accuracy=accuracy_score(labels, prediction), precision=precision,
                                 recall=recall, f1=f1, auc=roc_auc_score(labels, values),
                                 confusion_matrix=confusion_matrix(labels, prediction).tolist())
        output_path = args.output / f"{dataset.replace('/', '_')}.json"
        output_path.write_text(json.dumps(dict(paths=paths, labels=labels, scores=scores, metrics=metrics), indent=2))
        print(dataset, json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
