
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
SCRIPT = ROOT / "analysis" / "train_casanet_v3_rawterrain.py"
REPORT_DIR = ROOT / "outputs" / "report"
MODELS_DIR = ROOT / "models"
THRESHOLD = 0.50

source = SCRIPT.read_text()
prefix = source.split("benchmark_variants = {")[0]
ns = {}
exec(prefix, ns)

SylhetDataset = ns["SylhetDataset"]
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


test_ds = SylhetDataset('test', augment=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
model = CASANet(dropout_p=0.1)
state = torch.load(MODELS_DIR / 'casa_net_best_v3.pth', map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
y_true, y_prob = evaluate_prob(model, test_loader)
metrics = metrics_from_prob(y_true, y_prob, THRESHOLD)
metrics['Threshold'] = THRESHOLD
print(json.dumps(metrics, indent=2))
(REPORT_DIR / 'strict_casa_metrics_threshold_0_50.json').write_text(json.dumps(metrics, indent=2))
