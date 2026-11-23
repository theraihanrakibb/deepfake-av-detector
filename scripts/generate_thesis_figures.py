"""Build preview figures from the two protocol-specific experiment summaries."""

import argparse
import csv
import json
from pathlib import Path

MODELS = ("multimodal", "mesonet", "xceptionnet", "lipsync", "ensemble")
MODEL_LABELS = ("Multimodal", "MesoNet", "XceptionNet", "LipSync", "Ensemble")
SHORT_LABELS = ("Multi", "Meso", "Xcep", "Lip", "Ens")
ABLATIONS = (("Full", "Full"), ("V+A", "V+A"), ("V+T", "V+T"), ("A+T", "A+T"),
             ("Concat", "Concat"), ("Spatial", "Spatial"), ("No PE", "No PE"))
DATASETS = {"wdf": "WildDeepfake", "dfa": "DigiFakeAV"}
COLORS = {"wdf": "#2F6FA8", "dfa": "#C8781E", "real": "#2F6FA8", "fake": "#C8781E",
          "exp1": "#4A5A66", "exp2": "#1F7A62", "rgb": "#EAF1F8", "fft": "#F0ECF8",
          "audio": "#FDF2E9", "fusion": "#E7F4EE", "neutral": "#F4F6F7", "edge": "#33424D"}
CM = 1 / 2.54
SIZE_FULL = (13.5 * CM, 9 * CM)
SIZE_HALF = (9 * CM, 6.75 * CM)
np = None
plt = None
BOX_LABELS = []


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/local_preview/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/local_preview/tables"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def style():
    global np, plt
    import matplotlib as mpl
    import numpy
    import matplotlib.pyplot
    mpl.use("Agg")
    np = numpy
    plt = matplotlib.pyplot
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
        "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDE2E6", "grid.linewidth": 0.45, "grid.alpha": 0.7,
        "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "#DDE2E6",
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.edgecolor": "none",
    })


def load_json(path):
    return json.loads(Path(path).read_text())


def assert_text_inside_canvas(fig):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        extent = text.get_window_extent(renderer=renderer)
        if extent.x0 < -2 or extent.y0 < -2 or extent.x1 > fig.bbox.width + 2 or extent.y1 > fig.bbox.height + 2:
            raise AssertionError(f"text outside canvas: {text.get_text()!r}")


def fit_box_labels(fig, ax):
    for text, (x0, y0, x1, y1) in BOX_LABELS:
        bx0, by0 = ax.transData.transform((x0, y0))
        bx1, by1 = ax.transData.transform((x1, y1))
        lo_x, hi_x = min(bx0, bx1) + 2, max(bx0, bx1) - 2
        lo_y, hi_y = min(by0, by1) + 2, max(by0, by1) - 2
        size = float(text.get_fontsize())
        for _ in range(24):
            text.set_fontsize(size)
            fig.canvas.draw()
            extent = text.get_window_extent(renderer=fig.canvas.get_renderer())
            if extent.x0 >= lo_x and extent.x1 <= hi_x and extent.y0 >= lo_y and extent.y1 <= hi_y:
                break
            size -= 0.35
        else:
            raise AssertionError(f"box label does not fit: {text.get_text()!r}")
    BOX_LABELS.clear()


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    assert_text_inside_canvas(fig)
    fig.savefig(directory / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def box(ax, xy, w, h, label, face, fontsize=8.5):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=face, edgecolor=COLORS["edge"], linewidth=0.7))
    text = ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                   fontsize=fontsize, linespacing=1.12, color="#17202A")
    BOX_LABELS.append((text, (x + 0.08, y + 0.08, x + w - 0.08, y + h - 0.08)))


def arrow(ax, start, end, connection=None):
    props = {"arrowstyle": "-|>", "color": COLORS["edge"], "lw": 0.8, "shrinkA": 1, "shrinkB": 1}
    if connection:
        props["connectionstyle"] = connection
    ax.annotate("", end, start, arrowprops=props)


def diagram_canvas(size, xlim, ylim):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
    return fig, ax


def make_architecture(out):
    fig, ax = diagram_canvas(SIZE_FULL, (0, 12), (0, 6))
    box(ax, (0.20, 2.55), 1.45, 0.95, "Video clip\n30 frames", COLORS["rgb"])
    box(ax, (2.10, 3.75), 1.75, 0.95, "RGB\nEffNet-B0\n1280-d", COLORS["rgb"], 8.0)
    box(ax, (2.10, 1.55), 1.75, 0.95, "FFT\nEffNet-B0\n1280-d", COLORS["fft"], 8.0)
    box(ax, (4.35, 2.55), 1.45, 0.95, "Concat\n2560-d", COLORS["neutral"])
    box(ax, (6.30, 3.75), 1.75, 0.95, "Frame pool\n2560-d", COLORS["rgb"])
    box(ax, (6.30, 1.55), 1.75, 0.95, "Transformer\nCLS 512-d", COLORS["fft"])
    box(ax, (8.60, 2.55), 1.35, 0.95, "Attention\nfusion", COLORS["fusion"])
    box(ax, (10.45, 2.55), 1.35, 0.95, "Classifier\nfake score", COLORS["neutral"])
    box(ax, (0.20, 0.45), 1.45, 0.75, "Waveform\n16 kHz", COLORS["audio"], 8.0)
    box(ax, (2.10, 0.45), 1.75, 0.75, "MFCC\n40 x 157", COLORS["audio"], 8.0)
    box(ax, (4.35, 0.45), 1.45, 0.75, "Audio CNN\n512-d", COLORS["audio"], 8.0)
    for s, e in (((1.65, 3.02), (2.10, 4.22)), ((1.65, 3.02), (2.10, 2.02)),
                 ((3.85, 4.22), (4.35, 3.20)), ((3.85, 2.02), (4.35, 2.85)),
                 ((5.80, 3.02), (6.30, 4.22)), ((5.80, 3.02), (6.30, 2.02)),
                 ((8.05, 4.22), (8.60, 3.18)), ((8.05, 2.02), (8.60, 2.88)),
                 ((9.95, 3.02), (10.45, 3.02)), ((1.65, 0.82), (2.10, 0.82)),
                 ((3.85, 0.82), (4.35, 0.82))):
        arrow(ax, s, e)
    arrow(ax, (5.80, 0.82), (8.60, 2.70), "angle3,angleA=0,angleB=90")
    fit_box_labels(fig, ax)
    save(fig, out, "chapter3_architecture_overview")


def make_audio_pipeline(out):
    fig, ax = diagram_canvas(SIZE_FULL, (0, 12), (0, 4.2))
    stages = ((0.25, "Decode\nFFmpeg\n16 kHz mono"), (2.60, "Fix length\n5 s\n80,000 samples"),
              (4.95, "MFCC\n40 coefficients\n40 x 157"), (7.30, "Normalize\nper coefficient\nzero mean"),
              (9.65, "Audio CNN\nthree blocks\n512-d vector"))
    for x, label in stages:
        box(ax, (x, 2.35), 1.85, 1.10, label, COLORS["audio"], 8.0)
    for x in (2.10, 4.45, 6.80, 9.15):
        arrow(ax, (x, 2.90), (x + 0.50, 2.90))
    box(ax, (0.25, 0.55), 1.85, 0.85, "No audio\nstream", COLORS["neutral"], 8.0)
    box(ax, (2.60, 0.55), 1.85, 0.85, "Zero\nwaveform", COLORS["neutral"], 8.0)
    arrow(ax, (2.10, 0.98), (2.60, 0.98))
    arrow(ax, (4.45, 0.98), (5.10, 2.35), "angle3,angleA=0,angleB=90")
    fit_box_labels(fig, ax)
    save(fig, out, "chapter3_audio_processing_pipeline")


def make_temporal_context(out):
    fig, ax = diagram_canvas(SIZE_HALF, (0, 9), (0, 5.6))
    xs = np.linspace(0.8, 8.5, 10)
    for i, x in enumerate(xs):
        color = "#9FC5E8" if i in (1, 4, 7) else "#D9E8F5"
        ax.add_patch(plt.Rectangle((x - 0.20, 4.40), 0.40, 0.42, facecolor=color,
                                   edgecolor=COLORS["edge"], linewidth=0.5))
    for x, label in ((xs[1], "t1"), (xs[4], "t4"), (xs[7], "t7")):
        ax.text(x, 4.28, label, ha="center", va="top", fontsize=8.5)
    ax.text(4.65, 3.82, "30 uniformly sampled frames, order preserved", ha="center", fontsize=9.0)
    box(ax, (0.45, 2.10), 2.15, 0.90, "Linear proj.\n2560 to 512", COLORS["rgb"], 8.2)
    box(ax, (3.35, 2.10), 2.15, 0.90, "CLS token\n+ position", COLORS["fft"], 8.2)
    box(ax, (6.25, 2.10), 2.15, 0.90, "Transformer\n3 layers\n8 heads", COLORS["fusion"], 8.2)
    arrow(ax, (2.60, 2.55), (3.35, 2.55))
    arrow(ax, (5.50, 2.55), (6.25, 2.55))
    box(ax, (3.35, 0.55), 2.15, 0.75, "CLS vector\n512-d", COLORS["neutral"], 8.2)
    arrow(ax, (7.32, 2.10), (4.43, 1.30), "angle3,angleA=0,angleB=90")
    fit_box_labels(fig, ax)
    save(fig, out, "chapter3_temporal_context")


def make_attention_fusion(out):
    fig, ax = diagram_canvas(SIZE_HALF, (0, 9), (0, 5.2))
    for y, label, face in ((3.55, "Visual pool\n512-d", COLORS["rgb"]),
                           (2.20, "Temporal CLS\n512-d", COLORS["fft"]),
                           (0.85, "Audio CNN\n512-d", COLORS["audio"])):
        box(ax, (0.35, y), 2.15, 0.85, label, face, 8.2)
    box(ax, (3.45, 2.20), 1.65, 0.90, "Score\nsoftmax", COLORS["neutral"])
    box(ax, (5.75, 2.20), 1.25, 0.90, "Weights\nw1 w2 w3", COLORS["fusion"], 8.0)
    box(ax, (7.65, 2.20), 1.05, 0.90, "Fused\n512-d", COLORS["neutral"], 8.2)
    box(ax, (5.35, 0.45), 3.35, 0.75, "Classifier to one score", COLORS["neutral"], 8.2)
    for y in (4.00, 2.65, 1.30):
        arrow(ax, (2.50, y), (3.45, 2.75 if y == 4.00 else 2.65 if y == 2.65 else 2.55),
              "angle3,angleA=0,angleB=90" if y == 4.00 else None)
    arrow(ax, (5.10, 2.65), (5.75, 2.65))
    arrow(ax, (7.00, 2.65), (7.65, 2.65))
    arrow(ax, (8.18, 2.20), (7.00, 1.20), "angle3,angleA=0,angleB=90")
    fit_box_labels(fig, ax)
    save(fig, out, "chapter3_attention_fusion")


def make_experiment_protocol(out):
    fig, ax = diagram_canvas(SIZE_FULL, (0, 12), (0, 4.6))
    ax.text(0.35, 4.05, "Experiment 1", fontsize=10.5, weight="bold", color=COLORS["exp1"])
    ax.text(0.35, 2.05, "Experiment 2", fontsize=10.5, weight="bold", color=COLORS["exp2"])
    lanes = ((3.00, ("4 seeds\n20 epochs", "Val-loss\ncheckpoint", "Val-F1\ncutoff", "Equal\nensemble", "Held-out test"), "#EEF1F3"),
             (1.00, ("4 seeds\n20 epochs", "Val-AUC\ncheckpoint", "Val-F1\ncutoff", "Shrunk AUC\nensemble", "Held-out test"), "#E8F3EE"))
    for y, labels, face in lanes:
        xs = [0.35, 2.65, 4.95, 7.25, 9.55]
        for x, label in zip(xs, labels):
            box(ax, (x, y), 1.85, 0.85, label, face, 8.2)
        for i in range(4):
            arrow(ax, (xs[i] + 1.85, y + 0.43), (xs[i + 1], y + 0.43))
    fit_box_labels(fig, ax)
    save(fig, out, "chapter4_experiment_protocol")


def make_split_counts(out):
    labels = ["WDF train", "WDF val", "WDF test", "DFA train", "DFA val", "DFA test"]
    real, fake = [36, 8, 8, 70, 15, 15], [41, 9, 8, 140, 30, 30]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=SIZE_FULL)
    ax.bar(x, real, 0.62, color=COLORS["real"], label="Authentic")
    ax.bar(x, fake, 0.62, bottom=real, color=COLORS["fake"], label="Manipulated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Clips", labelpad=8)
    ax.set_ylim(0, 230)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.95)
    save(fig, out, "chapter4_dataset_split_counts")


def metric(data, ds, model, name):
    return data["datasets"][ds]["eval"][model][f"{name}_mean"]


def metric_std(data, ds, model, name):
    return data["datasets"][ds]["eval"][model].get(f"{name}_std", 0)


def make_experiment_auc(exp, out, stem):
    datasets = tuple(ds for ds in ("wdf", "dfa") if exp["datasets"][ds]["eval"])
    x = np.arange(len(MODELS))
    width = 0.34 if len(datasets) == 2 else 0.52
    fig, ax = plt.subplots(figsize=SIZE_FULL)
    for i, (ds, hatch) in enumerate(zip(datasets, ("", "///"))):
        offset = (i - (len(datasets) - 1) / 2) * (width + 0.04)
        ax.bar(x + offset, [metric(exp, ds, m, "auc") for m in MODELS], width,
               yerr=[metric_std(exp, ds, m, "auc") for m in MODELS], capsize=2.4,
               color=COLORS[ds], hatch=hatch, edgecolor=COLORS["edge"], linewidth=0.5,
               label=DATASETS[ds])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("ROC AUC", labelpad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.95)
    save(fig, out, stem)


def make_protocol_comparison(exp1, exp2, out):
    fig, axes = plt.subplots(1, 2, figsize=SIZE_FULL, sharey=True)
    x = np.arange(len(MODELS))
    for ax, ds in zip(axes, ("wdf", "dfa")):
        if ds == "wdf":
            for offset, exp, label, color in ((-0.20, exp1, "Experiment 1", COLORS["exp1"]),
                                              (0.20, exp2, "Experiment 2", COLORS["exp2"])):
                ax.bar(x + offset, [metric(exp, ds, m, "auc") for m in MODELS], 0.36,
                       yerr=[metric_std(exp, ds, m, "auc") for m in MODELS], capsize=2.0,
                       color=color, edgecolor=COLORS["edge"], linewidth=0.5, label=label)
        else:
            ax.bar(x, [metric(exp1, ds, m, "auc") for m in MODELS], 0.52,
                   yerr=[metric_std(exp1, ds, m, "auc") for m in MODELS], capsize=2.0,
                   color=COLORS["exp1"], edgecolor=COLORS["edge"], linewidth=0.5, label="Experiment 1")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=20, ha="right")
        ax.set_xlabel("WDF" if ds == "wdf" else "DFA", labelpad=6)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="upper left", fontsize=8.5)
    axes[0].set_ylabel("ROC AUC", labelpad=8)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.94, wspace=0.12)
    save(fig, out, "chapter4_protocol_comparison_auc")


def make_ablation(exp1, exp2, out):
    x = np.arange(len(ABLATIONS))
    fig, ax = plt.subplots(figsize=SIZE_FULL)
    for offset, ds, hatch in ((-0.20, "wdf", ""), (0.20, "dfa", "///")):
        source = exp2 if ds == "wdf" else exp1
        values = [source["datasets"][ds]["ablation"][short]["auc_mean"] for _, short in ABLATIONS]
        stds = [source["datasets"][ds]["ablation"][short].get("auc_std", 0) for _, short in ABLATIONS]
        ax.bar(x + offset, values, 0.36, yerr=stds, capsize=2.2, color=COLORS[ds], hatch=hatch,
               edgecolor=COLORS["edge"], linewidth=0.5, label=f"{DATASETS[ds]}, {'Exp 2' if ds == 'wdf' else 'Exp 1'}")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in ABLATIONS], rotation=12, ha="right")
    ax.set_ylabel("ROC AUC", labelpad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.95)
    save(fig, out, "chapter4_ablation_auc")


def make_multimodal_metrics(exp1, exp2, out):
    groups = [("WDF\nExp 1", exp1, "wdf"), ("DFA\nExp 1", exp1, "dfa"), ("WDF\nExp 2", exp2, "wdf")]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=SIZE_HALF)
    for i, (name, label, color) in enumerate((("accuracy", "Accuracy", "#4A5A66"), ("f1", "F1", "#C8781E"), ("auc", "ROC AUC", "#1F7A62"))):
        ax.bar(x + (i - 1) * 0.25, [metric(exp, ds, "multimodal", name) for _, exp, ds in groups], 0.25,
               yerr=[metric_std(exp, ds, "multimodal", name) for _, exp, ds in groups], capsize=2.0,
               color=color, edgecolor=COLORS["edge"], linewidth=0.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Score", labelpad=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.15, top=0.82)
    save(fig, out, "chapter4_multimodal_metrics")


def figure_plan():
    return [
        {"figure": "chapter3_architecture_overview.png", "nwpu_size": "13.5 x 9 cm", "caption": "Architecture of the multimodal detector. RGB and log FFT frames feed visual pooling and the temporal branch; MFCC features feed attention fusion.", "first_callout": "Chapter 3", "source": "verified source code", "status": "generated"},
        {"figure": "chapter3_audio_processing_pipeline.png", "nwpu_size": "13.5 x 9 cm", "caption": "Audio preprocessing and the silence fallback used for clips without an audio stream.", "first_callout": "Chapter 3", "source": "verified source code", "status": "generated"},
        {"figure": "chapter3_temporal_context.png", "nwpu_size": "9 x 6.75 cm", "caption": "Frame sampling, learned position and the Transformer context branch.", "first_callout": "Chapter 3", "source": "verified source code", "status": "generated"},
        {"figure": "chapter3_attention_fusion.png", "nwpu_size": "9 x 6.75 cm", "caption": "Attention fusion of visual, temporal and audio representations before classification.", "first_callout": "Chapter 3", "source": "verified source code", "status": "generated"},
        {"figure": "chapter4_experiment_protocol.png", "nwpu_size": "13.5 x 9 cm", "caption": "Training, checkpoint selection, cutoff selection and held-out measurement for Experiment 1 and Experiment 2, both at 4 seeds x 20 epochs.", "first_callout": "Chapter 4", "source": "saved run settings", "status": "generated"},
        {"figure": "chapter4_dataset_split_counts.png", "nwpu_size": "13.5 x 9 cm", "caption": "Authentic and manipulated clip counts in the training, validation and test partitions of both datasets.", "first_callout": "Chapter 4", "source": "recorded split counts", "status": "generated"},
        {"figure": "chapter4_experiment1_mean_auc.png", "nwpu_size": "13.5 x 9 cm", "caption": "Experiment 1 mean ROC AUC by detector and dataset. Error bars show one standard deviation across four seeds.", "first_callout": "Chapter 4", "source": "experiment_1_results.json", "status": "generated"},
        {"figure": "chapter4_experiment2_mean_auc.png", "nwpu_size": "13.5 x 9 cm", "caption": "Experiment 2 mean ROC AUC on WildDeepfake and DigiFakeAV.", "first_callout": "Chapter 4", "source": "experiment_2_results.json", "status": "generated"},
        {"figure": "chapter4_protocol_comparison_auc.png", "nwpu_size": "13.5 x 9 cm", "caption": "Mean ROC AUC by detector for both experiments and datasets.", "first_callout": "Chapter 4", "source": "experiment_1_results.json and experiment_2_results.json", "status": "generated"},
        {"figure": "chapter4_ablation_auc.png", "nwpu_size": "13.5 x 9 cm", "caption": "Mean ablation ROC AUC from the completed source protocol for each dataset. Error bars show one standard deviation.", "first_callout": "Chapter 4", "source": "experiment_1_results.json and experiment_2_results.json", "status": "generated"},
        {"figure": "chapter4_multimodal_metrics.png", "nwpu_size": "9 x 6.75 cm", "caption": "Accuracy, F1 and ROC AUC for the proposed multimodal detector under the completed protocols.", "first_callout": "Chapter 4", "source": "experiment_1_results.json and experiment_2_results.json", "status": "generated"},
    ]


def main():
    args = parse_args()
    if args.check:
        plan = figure_plan()
        assert len(plan) == 11
        assert all(row["figure"].startswith(("chapter3_", "chapter4_")) for row in plan)
        print("figure generator check: ok")
        return 0
    style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    exp1 = load_json(args.input_dir / "experiment_1_results.json")
    exp2 = load_json(args.input_dir / "experiment_2_results.json")
    make_architecture(args.figures_dir)
    make_audio_pipeline(args.figures_dir)
    make_temporal_context(args.figures_dir)
    make_attention_fusion(args.figures_dir)
    make_experiment_protocol(args.figures_dir)
    make_split_counts(args.figures_dir)
    make_experiment_auc(exp1, args.figures_dir, "chapter4_experiment1_mean_auc")
    make_experiment_auc(exp2, args.figures_dir, "chapter4_experiment2_mean_auc")
    make_protocol_comparison(exp1, exp2, args.figures_dir)
    make_ablation(exp1, exp2, args.figures_dir)
    make_multimodal_metrics(exp1, exp2, args.figures_dir)
    plan = figure_plan()
    fields = ("figure", "nwpu_size", "caption", "first_callout", "source", "status")
    with (args.tables_dir / "figure_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plan)
    audit = [{"figure": row["figure"], "caption": bool(row["caption"]), "first_callout": bool(row["first_callout"]),
              "source": bool(row["source"]), "png_exists": (args.figures_dir / row["figure"]).is_file(),
              "status": row["status"]} for row in plan]
    with (args.tables_dir / "figure_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("figure", "caption", "first_callout", "source", "png_exists", "status"))
        writer.writeheader()
        writer.writerows(audit)
    assert all((args.figures_dir / row["figure"]).is_file() for row in plan)
    print(f"Generated {len(plan)} preview figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
