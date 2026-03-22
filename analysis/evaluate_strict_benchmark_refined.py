
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
SCRIPT = ROOT / "analysis" / "train_benchmark_cuda_v2.py"
REPORT_DIR = ROOT / "outputs" / "report"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"

source = SCRIPT.read_text()
prefix = source.split("benchmark_variants = {")[0]
ns = {}
exec(prefix, ns)

SylhetDataset = ns["SylhetDataset"]
BaselineWrapper = ns["BaselineWrapper"]
CASANet = ns["CASANet"]
DataLoader = ns["DataLoader"]
metrics_from_prob = ns["metrics_from_prob"]
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


def sweep_refined(y_true, y_prob):
    rows = []
    for th in np.arange(0.20, 0.91, 0.02):
        rows.append({'threshold': float(th), **metrics_from_prob(y_true, y_prob, float(th))})
    df = pd.DataFrame(rows)
    best = df.sort_values(['IoU', 'F1', 'Dice'], ascending=False).iloc[0]
    return float(best['threshold']), df


val_ds = SylhetDataset('val', augment=False)
test_ds = SylhetDataset('test', augment=False)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

specs = [
    ('U-Net', BaselineWrapper('U-Net'), MODELS_DIR / 'benchmark_unet_v2.pth'),
    ('U-Net++', BaselineWrapper('U-Net++'), MODELS_DIR / 'benchmark_unetplusplus_v2.pth'),
    ('FPN', BaselineWrapper('FPN'), MODELS_DIR / 'benchmark_fpn_v2.pth'),
    ('DeepLabV3+', BaselineWrapper('DeepLabV3+'), MODELS_DIR / 'benchmark_deeplabv3plus_v2.pth'),
    ('MAnet', BaselineWrapper('MAnet'), MODELS_DIR / 'benchmark_manet_v2.pth'),
    ('CASA-Net', CASANet(dropout_p=0.1), MODELS_DIR / 'casa_net_best_v3.pth'),
]

rows = []
threshold_tables = []
for name, model, ckpt_path in specs:
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    val_true, val_prob = evaluate_prob(model, val_loader)
    test_true, test_prob = evaluate_prob(model, test_loader)
    th, th_df = sweep_refined(val_true, val_prob)
    test_metrics = metrics_from_prob(test_true, test_prob, th)
    row = {'Model': name, **test_metrics, 'Threshold': th, 'Params': int(sum(p.numel() for p in model.parameters()))}
    rows.append(row)
    local = th_df.copy()
    local.insert(0, 'Model', name)
    threshold_tables.append(local)
    print(name, json.dumps(row, indent=2))

benchmark_df = pd.DataFrame(rows).sort_values(['IoU', 'F1', 'Dice'], ascending=False).reset_index(drop=True)
benchmark_df['Rank'] = np.arange(1, len(benchmark_df) + 1)
benchmark_df.to_csv(REPORT_DIR / 'benchmark_results_best_refined.csv', index=False)
pd.concat(threshold_tables, ignore_index=True).to_csv(REPORT_DIR / 'strict_threshold_sweeps_refined.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(benchmark_df))
ax.bar(pos - 0.25, benchmark_df['IoU'], width=0.25, label='IoU', color='#264653')
ax.bar(pos, benchmark_df['F1'], width=0.25, label='F1', color='#2a9d8f')
ax.bar(pos + 0.25, benchmark_df['Precision'], width=0.25, label='Precision', color='#e76f51')
ax.set_xticks(pos)
ax.set_xticklabels(benchmark_df['Model'], rotation=15)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Strict benchmark with refined threshold sweep')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'benchmark_model_comparison_best_refined.png', dpi=300, bbox_inches='tight')
plt.close(fig)

summary = []
summary.append('# Strict Benchmark Summary With Refined Thresholds')
summary.append('')
summary.append(benchmark_df.to_markdown(index=False))
(REPORT_DIR / 'real_cuda_benchmark_summary_best_refined.md').write_text('\n'.join(summary), encoding='utf-8')
print(benchmark_df.to_string(index=False))
