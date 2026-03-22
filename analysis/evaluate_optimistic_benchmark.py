import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
ANALYSIS_SCRIPT = ROOT / "analysis" / "train_benchmark_cuda_optimistic.py"
REPORT_DIR = ROOT / "outputs" / "report"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"

source = ANALYSIS_SCRIPT.read_text()
prefix = source.split("benchmark_variants = {")[0]
ns = {}
exec(prefix, ns)

SylhetDataset = ns["SylhetDataset"]
BaselineWrapper = ns["BaselineWrapper"]
CASANet = ns["CASANet"]
metrics_from_prob = ns["metrics_from_prob"]
DataLoader = ns["DataLoader"]
DEVICE = ns["DEVICE"]


def evaluate_prob(model, loader):
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(DEVICE, non_blocking=True)
            vh = vh.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            pred = model(vv, vh, terrain).detach().cpu().numpy()
            y_prob.append(pred.reshape(-1))
            y_true.append(mask.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def best_history_metrics(history_path):
    hist = pd.read_csv(history_path)
    best = hist.sort_values(["val_iou", "val_f1", "val_dice"], ascending=False).iloc[0]
    return float(best["val_threshold"]), float(best["val_iou"]), int(best["epoch"])


def evaluate_checkpoint(name, model, ckpt_path, history_path, loader):
    threshold, best_val_iou, best_epoch = best_history_metrics(history_path)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    y_true, y_prob = evaluate_prob(model, loader)
    metrics = metrics_from_prob(y_true, y_prob, threshold)
    row = {
        "Model": name,
        **metrics,
        "Threshold": threshold,
        "BestValIoU": best_val_iou,
        "BestEpoch": best_epoch,
        "Params": int(sum(p.numel() for p in model.parameters())),
        "Checkpoint": str(ckpt_path),
    }
    return row


test_ds = SylhetDataset("test", augment=False)
test_loader = DataLoader(test_ds, batch_size=6, shuffle=False, num_workers=0, pin_memory=True)

specs = [
    (
        "U-Net",
        BaselineWrapper("U-Net"),
        MODELS_DIR / "benchmark_unet_optimistic.pth",
        REPORT_DIR / "unet_history_optimistic.csv",
    ),
    (
        "U-Net++",
        BaselineWrapper("U-Net++"),
        MODELS_DIR / "benchmark_unetplusplus_optimistic.pth",
        REPORT_DIR / "unetplusplus_history_optimistic.csv",
    ),
    (
        "FPN",
        BaselineWrapper("FPN"),
        MODELS_DIR / "benchmark_fpn_optimistic.pth",
        REPORT_DIR / "fpn_history_optimistic.csv",
    ),
    (
        "DeepLabV3+",
        BaselineWrapper("DeepLabV3+"),
        MODELS_DIR / "benchmark_deeplabv3plus_optimistic.pth",
        REPORT_DIR / "deeplabv3plus_history_optimistic.csv",
    ),
    (
        "MAnet",
        BaselineWrapper("MAnet"),
        MODELS_DIR / "benchmark_manet_optimistic.pth",
        REPORT_DIR / "manet_history_optimistic.csv",
    ),
    (
        "CASA-Net",
        CASANet(dropout_p=0.1),
        MODELS_DIR / "casa_net_best_optimistic.pth",
        REPORT_DIR / "casanet_history_optimistic.csv",
    ),
]

rows = []
for name, model, ckpt_path, history_path in specs:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {name}: {ckpt_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history for {name}: {history_path}")
    row = evaluate_checkpoint(name, model, ckpt_path, history_path, test_loader)
    rows.append(row)
    print(name, json.dumps({k: v for k, v in row.items() if k not in {"Model", "Checkpoint"}}, indent=2))

benchmark_df = pd.DataFrame(rows).sort_values(["IoU", "F1", "Dice"], ascending=False).reset_index(drop=True)
benchmark_df["Rank"] = np.arange(1, len(benchmark_df) + 1)
benchmark_df.to_csv(REPORT_DIR / "benchmark_results_optimistic.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(benchmark_df))
ax.bar(pos - 0.25, benchmark_df["IoU"], width=0.25, label="IoU", color="#264653")
ax.bar(pos, benchmark_df["F1"], width=0.25, label="F1", color="#2a9d8f")
ax.bar(pos + 0.25, benchmark_df["Dice"], width=0.25, label="Dice", color="#e76f51")
ax.set_xticks(pos)
ax.set_xticklabels(benchmark_df["Model"], rotation=15)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Optimistic overlap benchmark comparison")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "benchmark_model_comparison_optimistic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

leaderboard = benchmark_df.sort_values("Rank")
colors = ["#d62828" if m == "CASA-Net" else "#577590" for m in leaderboard["Model"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(leaderboard["Model"], leaderboard["IoU"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("IoU")
ax.set_title("Optimistic overlap leaderboard")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "benchmark_ranked_leaderboard_optimistic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

summary = []
summary.append("# Optimistic CUDA Benchmark Summary")
summary.append("")
summary.append("This benchmark uses an overlap-prone random patch split and is intentionally optimistic.")
summary.append("")
summary.append(benchmark_df.to_markdown(index=False))
summary.append("")
summary.append(f"Top model: {benchmark_df.iloc[0]['Model']}")
summary.append(f"Top IoU: {benchmark_df.iloc[0]['IoU']:.4f}")
summary.append(f"Top F1: {benchmark_df.iloc[0]['F1']:.4f}")
(REPORT_DIR / "real_cuda_benchmark_summary_optimistic.md").write_text("\n".join(summary), encoding="utf-8")

print(benchmark_df.to_string(index=False))
