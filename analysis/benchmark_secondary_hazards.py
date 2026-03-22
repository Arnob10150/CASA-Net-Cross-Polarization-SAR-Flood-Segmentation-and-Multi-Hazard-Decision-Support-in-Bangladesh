import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from rasterio.windows import Window
from scipy import ndimage as ndi
from sklearn.metrics import accuracy_score, average_precision_score, auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
PATCH_DIR = ROOT / "data" / "processed" / "patches_multihazard_ready"
PATCH_MANIFEST_PATH = ROOT / "data" / "processed" / "patches_full256" / "patch_manifest.csv"
SPLIT_INDEX_PATH = PATCH_DIR / "split_index.json"
NORM_STATS_PATH = PATCH_DIR / "norm_stats.json"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "report"
FIGURES_DIR = ROOT / "outputs" / "figures"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

for path in [MODELS_DIR, REPORT_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def normalize(arr, mean, std):
    return (arr - mean) / max(std, 1e-6)


def robust_norm(arr):
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


class HazardBenchmarkDataset(Dataset):
    def __init__(self, hazard="erosion", split="train", augment=False):
        self.hazard = hazard
        self.split = split
        self.augment = augment
        self.split_index = json.loads(SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
        self.paths = [Path(p) for p in self.split_index[split]]
        self.norm = json.loads(NORM_STATS_PATH.read_text(encoding="utf-8"))
        self.patch_manifest = pd.read_csv(PATCH_MANIFEST_PATH)
        self.lookup = {Path(path).name: (int(row), int(col)) for path, row, col in zip(self.patch_manifest["path"], self.patch_manifest["row"], self.patch_manifest["col"])}

        if hazard == "erosion":
            self.target_path = ROOT / "data" / "processed" / "masks_multihazard" / "erosion" / "erosion_seed_mask_sylhet_20240619.tif"
            self.context_path = ROOT / "data" / "processed" / "masks_multihazard" / "erosion" / "erosion_seed_mask_sylhet_20240619_v2.tif"
            self.start_ckpt = MODELS_DIR / "casa_net_erosion_context_v3.pth"
            self.supervision = "bangladesh_preferred_context_v3"
        elif hazard == "landslide":
            self.target_path = ROOT / "data" / "processed" / "masks_multihazard" / "landslide" / "landslide_seed_mask_sylhet_20240619_v2.tif"
            self.context_path = None
            self.start_ckpt = MODELS_DIR / "casa_net_landslide_v2.pth"
            self.supervision = "bangladesh_preferred_v2"
        else:
            raise ValueError(hazard)

        self.target_src = rasterio.open(self.target_path)
        self.context_src = rasterio.open(self.context_path) if self.context_path else None
        self.transform = A.Compose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)],
            additional_targets={"vh": "image", "t0": "image", "t1": "image", "t2": "image", "t3": "image", "mask": "mask"},
        )
        self.positive_px = []
        for path in self.paths:
            row, col = self.lookup[path.name]
            mask = self.target_src.read(1, window=Window(col, row, 256, 256))
            self.positive_px.append(float(mask.sum()))

    def __len__(self):
        return len(self.paths)

    def build_terrain(self, terrain, row, col):
        slope = terrain[0]
        twi = terrain[1]
        hand = terrain[3]
        if self.hazard == "erosion":
            edge = self.context_src.read(1, window=Window(col, row, 256, 256)).astype(np.float32)
            return np.stack([slope, twi, hand, edge], axis=0).astype(np.float32)
        mean = ndi.uniform_filter(slope, size=5)
        mean_sq = ndi.uniform_filter(slope * slope, size=5)
        rough = np.sqrt(np.clip(mean_sq - mean * mean, 0, None)).astype(np.float32)
        rough = robust_norm(rough)
        return np.stack([slope, hand, twi, rough], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        path = self.paths[idx]
        row, col = self.lookup[path.name]
        with np.load(path) as npz:
            vv = normalize(npz["vv"].astype(np.float32), self.norm["vv_mean"], self.norm["vv_std"])
            vh = normalize(npz["vh"].astype(np.float32), self.norm["vh_mean"], self.norm["vh_std"])
            terrain = npz["terrain"].astype(np.float32)
        terrain = self.build_terrain(terrain, row, col)
        mask = self.target_src.read(1, window=Window(col, row, 256, 256)).astype(np.float32)[None, ...]
        if self.augment:
            aug = self.transform(image=vv[0], vh=vh[0], t0=terrain[0], t1=terrain[1], t2=terrain[2], t3=terrain[3], mask=mask[0])
            vv = aug["image"][None, ...] + np.random.normal(0, 0.03, size=vv.shape).astype(np.float32)
            vh = aug["vh"][None, ...] + np.random.normal(0, 0.03, size=vh.shape).astype(np.float32)
            terrain = np.stack([aug["t0"], aug["t1"], aug["t2"], aug["t3"]], axis=0)
            mask = aug["mask"][None, ...]
        return torch.tensor(vv), torch.tensor(vh), torch.tensor(terrain), torch.tensor(mask)

    def sample_weights(self):
        weights = []
        total_px = 256 * 256
        for pos_px in self.positive_px:
            pct = pos_px / total_px
            if pct == 0:
                weight = 1.0
            elif pct < 0.005:
                weight = 3.5
            elif pct < 0.02:
                weight = 3.0
            elif pct < 0.1:
                weight = 2.5
            else:
                weight = 1.5
            weights.append(weight)
        return torch.DoubleTensor(weights)


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=3.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        bce = -(self.pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        return 0.55 * bce + 0.45 * dice.mean()


def metrics_from_prob(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "IoU": float(intersection / max(union, 1)),
        "Dice": float((2 * intersection) / max((y_true == 1).sum() + (y_pred == 1).sum(), 1)),
    }
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["ROC_AUC"] = float("nan")
    try:
        out["AveragePrecision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["AveragePrecision"] = float("nan")
    return out


def sweep_thresholds(y_true, y_prob):
    rows = []
    for threshold in np.arange(0.2, 0.86, 0.05):
        rows.append({"threshold": float(threshold), **metrics_from_prob(y_true, y_prob, float(threshold))})
    df = pd.DataFrame(rows)
    best = df.sort_values(["IoU", "F1", "Dice", "Precision"], ascending=False).iloc[0]
    return float(best["threshold"]), df


class CPAG(nn.Module):
    def __init__(self, vv_channels, vh_channels, inter_channels):
        super().__init__()
        self.q = nn.Conv2d(vv_channels, inter_channels, 1)
        self.k = nn.Conv2d(vh_channels, inter_channels, 1)
        self.v = nn.Conv2d(vh_channels, vh_channels, 1)
        self.proj = nn.Conv2d(vh_channels, vv_channels, 1)
        self.scale = inter_channels ** -0.5

    def forward(self, f_vv, g_vh):
        if g_vh.shape[-2:] != f_vv.shape[-2:]:
            g_vh = F.interpolate(g_vh, size=f_vv.shape[-2:], mode="bilinear", align_corners=False)
        q = self.q(f_vv).flatten(2)
        k = self.k(g_vh).flatten(2)
        v = self.v(g_vh).flatten(2)
        b, _, hw = q.shape
        side = int(hw ** 0.5)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, -1, side, side)
        out = self.proj(out)
        out = F.interpolate(out, size=f_vv.shape[-2:], mode="bilinear", align_corners=False)
        return f_vv + out


class HaorTerrainBranch(nn.Module):
    def __init__(self, in_channels=4, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.generators = nn.ModuleList([nn.Linear(64, 2 * channels) for channels in decoder_channels])

    def forward(self, terrain):
        context = self.encoder(terrain).flatten(1)
        return [generator(context) for generator in self.generators]


def apply_film(x, params):
    gamma, beta = params.chunk(2, dim=1)
    return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet34(weights=None)
        old_weight = backbone.conv1.weight.detach().clone()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:] = old_weight.mean(dim=1, keepdim=True)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(self.maxpool(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class MobileNetV3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(weights=None)
        old_weight = backbone.features[0][0].weight.detach().clone()
        backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            backbone.features[0][0].weight[:] = old_weight.mean(dim=1, keepdim=True)
        self.features = backbone.features

    def forward(self, x):
        outs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in {1, 3, 6, 11}:
                outs.append(x)
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(512, 256, 256), DecoderBlock(256, 128, 128), DecoderBlock(128, 64, 64)])
        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, fused, film_params):
        x = self.blocks[0](fused[-1], fused[-2])
        x = apply_film(x, film_params[0])
        x = self.blocks[1](x, fused[-3])
        x = apply_film(x, film_params[1])
        x = self.blocks[2](x, fused[-4])
        x = apply_film(x, film_params[2])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final(x)
        x = apply_film(x, film_params[3])
        return x


class ProbabilityHead(nn.Module):
    def __init__(self, in_channels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_p)
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(self.dropout(x)))


class CASANet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.vv_encoder = ResNet34Encoder()
        self.vh_encoder = MobileNetV3Encoder()
        self.cpag = nn.ModuleList([CPAG(64, 16, 32), CPAG(128, 24, 64), CPAG(256, 40, 128), CPAG(512, 96, 256)])
        self.terrain_branch = HaorTerrainBranch()
        self.decoder = UNetDecoder()
        self.head = ProbabilityHead(32, dropout_p)

    def forward(self, x_vv, x_vh, x_terrain):
        size = x_vv.shape[-2:]
        vv_feats = self.vv_encoder(x_vv)
        vh_feats = self.vh_encoder(x_vh)
        fused = [self.cpag[i](vv_feats[i], vh_feats[i]) for i in range(4)]
        film = self.terrain_branch(x_terrain)
        x = self.decoder(fused, film)
        x = self.head(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class SMPWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == "U-Net":
            self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, activation="sigmoid")
        elif model_name == "U-Net++":
            self.model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, activation="sigmoid")
        elif model_name == "FPN":
            self.model = smp.FPN(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, activation="sigmoid")
        elif model_name == "DeepLabV3+":
            self.model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, activation="sigmoid")
        elif model_name == "MAnet":
            self.model = smp.MAnet(encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=1, activation="sigmoid")
        else:
            raise ValueError(model_name)

    def forward(self, x_vv, x_vh, _x_terrain):
        return self.model(torch.cat([x_vv, x_vh], dim=1))


def count_params(model):
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def collect_probs(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(device, non_blocking=True)
            vh = vh.to(device, non_blocking=True)
            terrain = terrain.to(device, non_blocking=True)
            pred = model(vv, vh, terrain)
            probs.append(pred.detach().cpu().numpy().ravel())
            targets.append(mask.numpy().ravel())
    return np.concatenate(targets).astype(np.uint8), np.concatenate(probs)


def evaluate_epoch(model, loader, criterion, threshold, device):
    model.eval()
    losses, probs, targets = [], [], []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(device, non_blocking=True)
            vh = vh.to(device, non_blocking=True)
            terrain = terrain.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(vv, vh, terrain)
            losses.append(float(criterion(pred, mask).item()))
            probs.append(pred.detach().cpu().numpy().ravel())
            targets.append(mask.detach().cpu().numpy().ravel())
    y_true = np.concatenate(targets).astype(np.uint8)
    y_prob = np.concatenate(probs)
    metrics = metrics_from_prob(y_true, y_prob, threshold)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_model(model_name, model, train_loader, val_loader, test_loader, criterion, device, epochs, patience, hazard, start_ckpt=None):
    if start_ckpt is not None and Path(start_ckpt).exists():
        state = torch.load(start_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5 if model_name == "CASA-Net" else 1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=True)
    best_state = deepcopy(model.state_dict())
    best_iou = -1.0
    best_threshold = 0.5
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for vv, vh, terrain, mask in train_loader:
            vv = vv.to(device, non_blocking=True)
            vh = vh.to(device, non_blocking=True)
            terrain = terrain.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                pred = model(vv, vh, terrain)
                loss = criterion(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
        scheduler.step()

        y_val_true, y_val_prob = collect_probs(model, val_loader, device)
        threshold, _ = sweep_thresholds(y_val_true, y_val_prob)
        val_metrics = metrics_from_prob(y_val_true, y_val_prob, threshold)
        val_loss = evaluate_epoch(model, val_loader, criterion, threshold, device)["loss"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_threshold": threshold, **val_metrics})
        print(json.dumps({"hazard": hazard, "model": model_name, "epoch": epoch, "val_iou": round(val_metrics["IoU"], 4), "val_f1": round(val_metrics["F1"], 4), "val_precision": round(val_metrics["Precision"], 4), "val_recall": round(val_metrics["Recall"], 4), "threshold": threshold}))
        if val_metrics["IoU"] > best_iou:
            best_iou = val_metrics["IoU"]
            best_threshold = threshold
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    y_true, y_prob = collect_probs(model, test_loader, device)
    test_metrics = metrics_from_prob(y_true, y_prob, best_threshold)
    test_metrics["threshold"] = best_threshold
    test_metrics["Model"] = model_name
    test_metrics["Params"] = count_params(model)
    test_metrics["hazard"] = hazard
    test_metrics["duration_sec"] = time.time() - start

    history_df = pd.DataFrame(history)
    model_slug = model_name.lower().replace("+", "plus").replace("-", "").replace(" ", "_")
    history_path = REPORT_DIR / f"{hazard}_{model_slug}_secondary_history.csv"
    history_df.to_csv(history_path, index=False)
    ckpt_path = MODELS_DIR / f"{hazard}_{model_slug}_secondary_benchmark.pth"
    torch.save(model.state_dict(), ckpt_path)

    summary = {"hazard": hazard, "model": model_name, "checkpoint": str(ckpt_path), "history_csv": str(history_path), "best_threshold": best_threshold, **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()}}
    (REPORT_DIR / f"{hazard}_{model_slug}_secondary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return test_metrics, y_true, y_prob, history_df


def save_branch_figures(hazard, benchmark_df, casanet_y_true, casanet_y_prob, casanet_history):
    benchmark_df = benchmark_df.sort_values(["IoU", "F1", "Precision"], ascending=False).reset_index(drop=True)
    benchmark_df["Rank"] = np.arange(1, len(benchmark_df) + 1)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plot_df = benchmark_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1", "IoU"], var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"{hazard.title()} benchmark metrics across models")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_benchmark_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.barplot(data=benchmark_df, x="IoU", y="Model", palette="viridis", ax=ax)
    ax.set_xlim(0, max(0.4, float(benchmark_df["IoU"].max()) + 0.05))
    ax.set_title(f"{hazard.title()} ranked leaderboard (IoU)")
    for idx, value in enumerate(benchmark_df["IoU"]):
        ax.text(float(value) + 0.005, idx, f"{value:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_benchmark_ranked_leaderboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    heat_df = benchmark_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "IoU", "ROC_AUC", "AveragePrecision"]]
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{hazard.title()} model comparison heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_model_comparison_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    threshold = float(benchmark_df.loc[benchmark_df["Model"] == "CASA-Net", "threshold"].iloc[0])
    y_pred = (casanet_y_prob >= threshold).astype(np.uint8)
    cm = confusion_matrix(casanet_y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{hazard.title()} CASA-Net confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_confusion_matrix_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    precision_vals, recall_vals, _ = precision_recall_curve(casanet_y_true, casanet_y_prob)
    fpr, tpr, _ = roc_curve(casanet_y_true, casanet_y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    axes[0].plot(recall_vals, precision_vals, color="#1d3557", linewidth=2)
    axes[0].set_title(f"{hazard.title()} CASA-Net precision-recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].plot(fpr, tpr, color="#e76f51", linewidth=2, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].legend()
    axes[1].set_title(f"{hazard.title()} CASA-Net ROC")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_precision_recall_roc_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(casanet_history["epoch"], casanet_history["train_loss"], label="Train loss", color="#1d3557")
    ax.plot(casanet_history["epoch"], casanet_history["val_loss"], label="Val loss", color="#e76f51")
    ax2 = ax.twinx()
    ax2.plot(casanet_history["epoch"], casanet_history["IoU"], label="Val IoU", color="#2a9d8f")
    ax.set_title(f"{hazard.title()} CASA-Net loss and IoU curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("IoU")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{hazard}_loss_accuracy_curve_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_maturity_outputs(branch_rows):
    new_rows = pd.DataFrame(branch_rows)
    branch_path = REPORT_DIR / "secondary_hazard_benchmark_summary.csv"
    new_rows.to_csv(branch_path, index=False)

    current_path = REPORT_DIR / "multi_hazard_branch_metrics.csv"
    if current_path.exists():
        current = pd.read_csv(current_path)
        for _, row in new_rows.iterrows():
            current = current[current["hazard"] != row["hazard"]]
            current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
        current.to_csv(current_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plot_df = new_rows.melt(id_vars=["hazard", "best_model"], value_vars=["Accuracy", "Precision", "Recall", "F1", "IoU"], var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="hazard", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Secondary hazard branch metrics after benchmark upgrade")
    ax.legend(title="Hazard")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "secondary_hazard_branch_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazard", required=True, choices=["erosion", "landslide"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["U-Net", "U-Net++", "FPN", "DeepLabV3+", "MAnet", "CASA-Net"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available")
    print("Using device:", device, torch.cuda.get_device_name(0))

    train_ds = HazardBenchmarkDataset(args.hazard, "train", augment=True)
    val_ds = HazardBenchmarkDataset(args.hazard, "val", augment=False)
    test_ds = HazardBenchmarkDataset(args.hazard, "test", augment=False)
    train_sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_ratio = sum(train_ds.positive_px) / (len(train_ds) * 256 * 256)
    pos_weight = max(1.0, (1 - train_ratio) / max(train_ratio, 1e-6))
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight)
    print(json.dumps({"hazard": args.hazard, "train": len(train_ds), "val": len(val_ds), "test": len(test_ds), "train_ratio": train_ratio, "pos_weight": pos_weight}))

    model_builders = {
        "U-Net": lambda: SMPWrapper("U-Net"),
        "U-Net++": lambda: SMPWrapper("U-Net++"),
        "FPN": lambda: SMPWrapper("FPN"),
        "DeepLabV3+": lambda: SMPWrapper("DeepLabV3+"),
        "MAnet": lambda: SMPWrapper("MAnet"),
        "CASA-Net": lambda: CASANet(dropout_p=0.1),
    }
    results = []
    casanet_payload = None
    for model_name in args.models:
        metrics, y_true, y_prob, history_df = train_one_model(
            model_name=model_name,
            model=model_builders[model_name](),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            hazard=args.hazard,
            start_ckpt=train_ds.start_ckpt if model_name == "CASA-Net" else None,
        )
        results.append(metrics)
        if model_name == "CASA-Net":
            casanet_payload = (y_true, y_prob, history_df)

    benchmark_df = pd.DataFrame(results).sort_values(["IoU", "F1", "Precision"], ascending=False).reset_index(drop=True)
    benchmark_df["Rank"] = np.arange(1, len(benchmark_df) + 1)
    benchmark_df["supervision_type"] = train_ds.supervision
    benchmark_df["split_file"] = str(SPLIT_INDEX_PATH)
    benchmark_path = REPORT_DIR / f"benchmark_results_{args.hazard}_secondary.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    if casanet_payload is not None:
        save_branch_figures(args.hazard, benchmark_df, *casanet_payload)

    best_row = benchmark_df.iloc[0].to_dict()
    branch_row = {
        "hazard": args.hazard,
        "best_model": best_row["Model"],
        "Accuracy": best_row["Accuracy"],
        "Precision": best_row["Precision"],
        "Recall": best_row["Recall"],
        "F1": best_row["F1"],
        "IoU": best_row["IoU"],
        "Dice": best_row["Dice"],
        "ROC_AUC": best_row.get("ROC_AUC", float("nan")),
        "AveragePrecision": best_row.get("AveragePrecision", float("nan")),
        "threshold": best_row["threshold"],
        "supervision_type": train_ds.supervision,
        "benchmark_csv": str(benchmark_path),
    }
    update_maturity_outputs([branch_row])

    summary_lines = [
        f"# {args.hazard.title()} Secondary Benchmark",
        "",
        f"- Supervision: `{train_ds.supervision}`",
        f"- Split file: `{SPLIT_INDEX_PATH}`",
        f"- Train/val/test patches: `{len(train_ds)}/{len(val_ds)}/{len(test_ds)}`",
        f"- Best model: `{best_row['Model']}`",
        f"- Accuracy: `{best_row['Accuracy']:.4f}`",
        f"- Precision: `{best_row['Precision']:.4f}`",
        f"- Recall: `{best_row['Recall']:.4f}`",
        f"- F1: `{best_row['F1']:.4f}`",
        f"- IoU: `{best_row['IoU']:.4f}`",
        f"- ROC-AUC: `{best_row.get('ROC_AUC', float('nan')):.4f}`",
        f"- Average Precision: `{best_row.get('AveragePrecision', float('nan')):.4f}`",
        "",
        f"Saved table: `{benchmark_path}`",
    ]
    (REPORT_DIR / f"{args.hazard}_secondary_benchmark_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"hazard": args.hazard, "benchmark_csv": str(benchmark_path), "best_model": best_row["Model"], "IoU": best_row["IoU"], "F1": best_row["F1"]}, indent=2))


if __name__ == "__main__":
    main()
