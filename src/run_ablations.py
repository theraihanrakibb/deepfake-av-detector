"""Train and evaluate controlled MultimodalModel ablations."""

import os

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.datasets import DeepfakeDataset, load_split_manifest, split_dataset
from models.multimodal_model import MultimodalModel
from evaluate_detectors import evaluate_model
from train_detectors import collect_video_files, train_multimodal_model, make_balanced_sampler

ABLATION_CONFIGS = [
    ('Full (V+A+T, attn, FFT)', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('V+A (no temporal)', {
        'streams': ('visual', 'audio'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('V+T (no audio)', {
        'streams': ('visual', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('A+T (no visual fusion)', {
        'streams': ('audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('Concat fusion', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'concat',
        'use_frequency': True,
    }),
    ('Spatial-only (no FFT)', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': False,
    }),
    ('No positional encoding', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
        'use_positional': False,
    }),
]


def run_single_ablation(name, model_kwargs, train_paths, train_labels,
                        test_paths, test_labels, args, device):
    dataset_kwargs = dict(require_audio=args.require_audio,
                          skip_face_crop=args.skip_face_crop)
    train_dataset = DeepfakeDataset(video_paths=train_paths, labels=train_labels,
                                    augment=True, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_labels) if len(set(train_labels)) == 2 else None,
        shuffle=(len(set(train_labels)) != 2),
    )
    test_dataset = DeepfakeDataset(video_paths=test_paths, labels=test_labels,
                                   augment=False, **dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultimodalModel(pretrained=args.pretrained, **model_kwargs).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if not args.no_train:
        for epoch in range(args.epochs):
            train_loss = train_multimodal_model(model, train_loader, optimizer, loss_fn, device)
            print(f"  Epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f}")
    else:
        print('  Skipping training (--no_train)')

    metrics, _, _ = evaluate_model(model, test_loader, device)
    metrics['config'] = name
    print(f"  Results: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run controlled ablation studies')
    parser.add_argument('--data_dir', type=str, default='tests/test_data')
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results/2026-11-22/ablations'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--require-audio', action='store_true')
    parser.add_argument('--skip-face-crop', action='store_true')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.split_manifest:
        splits = load_split_manifest(args.split_manifest)
        train_paths, train_labels = map(list, zip(*splits['train']))
        test_paths, test_labels = map(list, zip(*splits['test']))
    else:
        video_files, labels = collect_video_files(args.data_dir, args.max_samples)
        if not video_files:
            print('No video files found.')
            return
        has_both_classes = len(set(int(label) for label in labels)) == 2
        if has_both_classes and len(video_files) >= 4:
            train_paths, train_labels, _, _, test_paths, test_labels = split_dataset(
                video_files, labels, ratios=(0.7, 0.0, 0.3), seed=args.seed
            )
        else:
            train_paths, train_labels = video_files, labels
            test_paths, test_labels = video_files, labels
            print('Warning: insufficient data for proper split; using same data for train/test')
    print(f'Train: {len(train_paths)} samples | Test: {len(test_paths)} samples')

    all_results = []
    for name, model_kwargs in ABLATION_CONFIGS:
        try:
            metrics = run_single_ablation(
                name, model_kwargs, train_paths, train_labels,
                test_paths, test_labels, args, device
            )
            all_results.append(metrics)
        except Exception as exc:
            print(f'  FAILED: {exc}')
            all_results.append({'config': name, 'accuracy': float('nan'),
                                'precision': float('nan'), 'recall': float('nan'),
                                'f1': float('nan'), 'auc': float('nan')})
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    df_thesis = df.rename(columns={
        'config': 'Model', 'accuracy': 'Accuracy', 'precision': 'Precision',
        'recall': 'Recall', 'f1': 'F1', 'auc': 'AUC',
    })
    for column in ('Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'):
        if column not in df_thesis.columns:
            df_thesis[column] = float('nan')
    df_thesis = df_thesis[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
    csv_path = args.output_dir / 'ablation_results.csv'
    df_thesis.to_csv(str(csv_path), index=False)
    print(f'\nResults saved to {csv_path}')
    print(df_thesis.to_string(index=False))

    from plot_thesis_results import _color_for, performance_comparison
    performance_comparison(str(csv_path), str(args.output_dir / 'ablation_comparison.png'))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(df_thesis) + 1)))
    baseline_f1 = df_thesis.iloc[0]['F1']
    if baseline_f1 > 0:
        df_rel = df_thesis.assign(relative=df_thesis['F1'] / baseline_f1)
        metric_label = 'Relative F1 (Full model = 1.0)'
    else:
        baseline_acc = df_thesis.iloc[0]['Accuracy']
        df_rel = df_thesis.assign(relative=df_thesis['Accuracy'] / baseline_acc if baseline_acc > 0 else 1.0)
        metric_label = 'Relative Accuracy (Full model = 1.0)' if baseline_acc > 0 else 'No variation (all zero)'
    df_rel = df_rel.sort_values('relative', ascending=True)
    colors = [_color_for(model) for model in df_rel['Model']]
    bars = ax.barh(df_rel['Model'], df_rel['relative'], color=colors,
                   edgecolor='white', linewidth=1)
    for bar, value in zip(bars, df_rel['relative']):
        ax.annotate(f'{value:.3f}x', xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=10, fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Full model baseline')
    ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title('Ablation Impact (Relative to Full Model)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    rel_path = args.output_dir / 'ablation_relative_impact.png'
    plt.savefig(str(rel_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Figure saved to {rel_path}')


if __name__ == '__main__':
    main()
