import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import rasterio
from rasterio.windows import Window

ROOT = Path(r'f:\MAPATHON\sylhet_flood_2024')
PATCH_DIR = ROOT / 'data' / 'processed' / 'patches_overlap128_optimistic'
MODELS_DIR = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'
REPORT_DIR = ROOT / 'outputs' / 'report'
SPLIT_INDEX_PATH = PATCH_DIR / 'split_index.json'
NORM_STATS_PATH = PATCH_DIR / 'norm_stats.json'
PATCH_MANIFEST_PATH = PATCH_DIR / 'patch_manifest.csv'
TARGET_PATH = ROOT / 'data' / 'processed' / 'masks_multihazard' / 'erosion' / 'erosion_fused_active_bank_zone_sylhet_v1.tif'
EDGE_CONTEXT_PATH = ROOT / 'data' / 'processed' / 'masks_multihazard' / 'erosion' / 'erosion_seed_mask_sylhet_20240619_v2.tif'
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def normalize(arr, mean, std):
    return (arr - mean) / max(std, 1e-6)


class ErosionContextDataset(Dataset):
    def __init__(self, split='train', augment=False):
        self.split = split
        self.augment = augment
        self.split_index = json.loads(SPLIT_INDEX_PATH.read_text())
        self.paths = [Path(p) for p in self.split_index[split]]
        self.norm = json.loads(NORM_STATS_PATH.read_text())
        self.patch_manifest = pd.read_csv(PATCH_MANIFEST_PATH)
        self.lookup = {Path(p).name: (int(r), int(c)) for p, r, c in zip(self.patch_manifest['path'], self.patch_manifest['row'], self.patch_manifest['col'])}
        self.target_src = rasterio.open(TARGET_PATH)
        self.context_src = rasterio.open(EDGE_CONTEXT_PATH)
        self.transform = A.Compose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)],
            additional_targets={'vh': 'image', 't0': 'image', 't1': 'image', 't2': 'image', 't3': 'image', 'mask': 'mask'}
        )
        self.positive_px = []
        for path in self.paths:
            row, col = self.lookup[path.name]
            mask = self.target_src.read(1, window=Window(col, row, 256, 256))
            self.positive_px.append(float(mask.sum()))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        row, col = self.lookup[path.name]
        with np.load(path) as npz:
            vv = normalize(npz['vv'].astype(np.float32), self.norm['vv_mean'], self.norm['vv_std'])
            vh = normalize(npz['vh'].astype(np.float32), self.norm['vh_mean'], self.norm['vh_std'])
            terrain = npz['terrain'].astype(np.float32)
        slope = terrain[0]
        twi = terrain[1]
        hand = terrain[3]
        edge = self.context_src.read(1, window=Window(col, row, 256, 256)).astype(np.float32)
        terrain = np.stack([slope, twi, hand, edge], axis=0).astype(np.float32)
        mask = self.target_src.read(1, window=Window(col, row, 256, 256)).astype(np.float32)[None, ...]
        if self.augment:
            aug = self.transform(image=vv[0], vh=vh[0], t0=terrain[0], t1=terrain[1], t2=terrain[2], t3=terrain[3], mask=mask[0])
            vv = aug['image'][None, ...] + np.random.normal(0, 0.03, size=vv.shape).astype(np.float32)
            vh = aug['vh'][None, ...] + np.random.normal(0, 0.03, size=vh.shape).astype(np.float32)
            terrain = np.stack([aug['t0'], aug['t1'], aug['t2'], aug['t3']], axis=0)
            mask = aug['mask'][None, ...]
        return torch.tensor(vv), torch.tensor(vh), torch.tensor(terrain), torch.tensor(mask)

    def sample_weights(self):
        weights = []
        total_px = 256 * 256
        for pos_px in self.positive_px:
            pct = pos_px / total_px
            if pct == 0:
                w = 1.0
            elif pct < 0.01:
                w = 2.0
            elif pct < 0.05:
                w = 3.0
            elif pct < 0.2:
                w = 2.5
            else:
                w = 1.5
            weights.append(w)
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
        return 0.6 * bce + 0.4 * dice.mean()


def metrics_from_prob(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    out = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'IoU': float(intersection / max(union, 1)),
        'Dice': float((2 * intersection) / max((y_true == 1).sum() + (y_pred == 1).sum(), 1)),
        'ROC_AUC': float(roc_auc_score(y_true, y_prob)),
        'AveragePrecision': float(average_precision_score(y_true, y_prob)),
    }
    return out


def sweep_thresholds(y_true, y_prob):
    rows = []
    for th in np.arange(0.2, 0.81, 0.05):
        rows.append({'threshold': float(th), **metrics_from_prob(y_true, y_prob, float(th))})
    df = pd.DataFrame(rows)
    best = df.sort_values(['IoU', 'F1', 'Dice', 'Precision'], ascending=False).iloc[0]
    return float(best['threshold']), df


class CPAG(nn.Module):
    def __init__(self, vv_channels, vh_channels, inter_channels):
        super().__init__(); self.Q = nn.Conv2d(vv_channels, inter_channels, 1); self.K = nn.Conv2d(vh_channels, inter_channels, 1); self.V = nn.Conv2d(vh_channels, vh_channels, 1); self.proj = nn.Conv2d(vh_channels, vv_channels, 1); self.scale = inter_channels ** -0.5
    def forward(self, f_vv, g_vh):
        if g_vh.shape[-2:] != f_vv.shape[-2:]:
            g_vh = F.interpolate(g_vh, size=f_vv.shape[-2:], mode='bilinear', align_corners=False)
        q = self.Q(f_vv).flatten(2); k = self.K(g_vh).flatten(2); v = self.V(g_vh).flatten(2); b, _, hw = q.shape; side = int(hw ** 0.5)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, -1, side, side)
        out = self.proj(out)
        out = F.interpolate(out, size=f_vv.shape[-2:], mode='bilinear', align_corners=False)
        return f_vv + out

class HaorTerrainBranch(nn.Module):
    def __init__(self, in_channels=4, decoder_channels=(256, 128, 64, 32)):
        super().__init__(); self.encoder = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)); self.generators = nn.ModuleList([nn.Linear(64, 2 * c) for c in decoder_channels])
    def forward(self, terrain):
        context = self.encoder(terrain).flatten(1); return [layer(context) for layer in self.generators]

def apply_film(x, params):
    gamma, beta = params.chunk(2, dim=1); return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__(); backbone = torchvision.models.resnet34(weights=None); backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False); self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu); self.maxpool = backbone.maxpool; self.layer1 = backbone.layer1; self.layer2 = backbone.layer2; self.layer3 = backbone.layer3; self.layer4 = backbone.layer4
    def forward(self, x):
        x = self.stem(x); f1 = self.layer1(self.maxpool(x)); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3); return [f1, f2, f3, f4]

class MobileNetV3Encoder(nn.Module):
    def __init__(self):
        super().__init__(); backbone = torchvision.models.mobilenet_v3_small(weights=None); backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False); self.features = backbone.features
    def forward(self, x):
        outs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in {1, 3, 6, 11}: outs.append(x)
        return outs

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__(); self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1); self.bn1 = nn.BatchNorm2d(out_channels); self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1); self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False); x = torch.cat([x, skip], dim=1); x = F.relu(self.bn1(self.conv1(x)), inplace=True); x = F.relu(self.bn2(self.conv2(x)), inplace=True); return x

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList([DecoderBlock(512, 256, 256), DecoderBlock(256, 128, 128), DecoderBlock(128, 64, 64)]); self.final = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
    def forward(self, fused, film_params):
        x = self.blocks[0](fused[-1], fused[-2]); x = apply_film(x, film_params[0]); x = self.blocks[1](x, fused[-3]); x = apply_film(x, film_params[1]); x = self.blocks[2](x, fused[-4]); x = apply_film(x, film_params[2]); x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False); x = self.final(x); x = apply_film(x, film_params[3]); return x

class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, dropout_p=0.1):
        super().__init__(); self.dropout = nn.Dropout2d(dropout_p); self.conv = nn.Conv2d(in_channels, 1, 1)
    def forward(self, x):
        return torch.sigmoid(self.conv(self.dropout(x)))

class CASANet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__(); self.vv_encoder = ResNet34Encoder(); self.vh_encoder = MobileNetV3Encoder(); self.cpag = nn.ModuleList([CPAG(64, 16, 32), CPAG(128, 24, 64), CPAG(256, 40, 128), CPAG(512, 96, 256)]); self.terrain_branch = HaorTerrainBranch(); self.decoder = UNetDecoder(); self.head = UncertaintyHead(32, dropout_p)
    def forward(self, x_vv, x_vh, x_terrain):
        size = x_vv.shape[-2:]; vv_feats = self.vv_encoder(x_vv); vh_feats = self.vh_encoder(x_vh); fused = [self.cpag[i](vv_feats[i], vh_feats[i]) for i in range(4)]; film = self.terrain_branch(x_terrain); x = self.decoder(fused, film); x = self.head(x); return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def collect_probs(model, loader, device):
    model.eval(); probs = []; targets = []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(device, non_blocking=True); vh = vh.to(device, non_blocking=True); terrain = terrain.to(device, non_blocking=True); pred = model(vv, vh, terrain); probs.append(pred.cpu().numpy().ravel()); targets.append(mask.numpy().ravel())
    return np.concatenate(targets).astype(np.uint8), np.concatenate(probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--epochs', type=int, default=8); parser.add_argument('--batch-size', type=int, default=6); parser.add_argument('--patience', type=int, default=4); args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('CUDA device not available')
    print('Using device:', device, torch.cuda.get_device_name(0))

    train_ds = ErosionContextDataset('train', augment=True); val_ds = ErosionContextDataset('val', augment=False); test_ds = ErosionContextDataset('test', augment=False)
    train_sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_ratio = sum(train_ds.positive_px) / (len(train_ds) * 256 * 256)
    pos_weight = max(1.0, (1 - train_ratio) / max(train_ratio, 1e-6))
    print({'train_ratio': train_ratio, 'pos_weight': pos_weight, 'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)})

    model = CASANet(dropout_p=0.1).to(device)
    start_ckpt = MODELS_DIR / 'erosion_optimistic_casanet_secondary_benchmark_ft.pth'
    model.load_state_dict(torch.load(start_ckpt, map_location='cpu'), strict=False)
    print('Loaded start weights from', start_ckpt.name)

    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state = deepcopy(model.state_dict()); best_iou = -1.0; best_threshold = 0.5; wait = 0; history = []; start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for vv, vh, terrain, mask in train_loader:
            vv = vv.to(device, non_blocking=True); vh = vh.to(device, non_blocking=True); terrain = terrain.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); pred = model(vv, vh, terrain); loss = criterion(pred, mask); loss.backward(); optimizer.step(); train_losses.append(float(loss.item()))
        scheduler.step()
        y_val_true, y_val_prob = collect_probs(model, val_loader, device)
        threshold, _ = sweep_thresholds(y_val_true, y_val_prob)
        val_metrics = metrics_from_prob(y_val_true, y_val_prob, threshold)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'val_threshold': threshold, **val_metrics})
        print({'epoch': epoch, 'val_iou': round(val_metrics['IoU'], 4), 'val_f1': round(val_metrics['F1'], 4), 'val_precision': round(val_metrics['Precision'], 4), 'val_recall': round(val_metrics['Recall'], 4), 'threshold': threshold})
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']; best_threshold = threshold; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
        if wait >= args.patience: break

    model.load_state_dict(best_state); ckpt_path = MODELS_DIR / 'casa_net_erosion_fused_active_bank_zone_v1.pth'; torch.save(model.state_dict(), ckpt_path)
    history_df = pd.DataFrame(history); history_path = REPORT_DIR / 'erosion_fused_active_bank_zone_v1_history.csv'; history_df.to_csv(history_path, index=False)
    y_true, y_prob = collect_probs(model, test_loader, device); test_metrics = metrics_from_prob(y_true, y_prob, best_threshold); test_metrics['threshold'] = best_threshold
    metrics_path = REPORT_DIR / 'test_metrics_erosion_fused_active_bank_zone_v1.csv'; pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    fig, ax = plt.subplots(figsize=(5,4), dpi=300); sns.heatmap(confusion_matrix(y_true, (y_prob>=best_threshold).astype(np.uint8)), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax); ax.set_title('Erosion fused active bank zone confusion matrix'); plt.tight_layout(); plt.savefig(FIGURES_DIR / 'erosion_confusion_matrix_fused_active_bank_zone_v1.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob); fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, axes = plt.subplots(1,2, figsize=(12,4), dpi=300); axes[0].plot(recall_vals, precision_vals); axes[0].set_title('Erosion fused active bank zone PR'); axes[1].plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.3f}'); axes[1].plot([0,1],[0,1],'--',color='gray'); axes[1].legend(); axes[1].set_title('Erosion fused active bank zone ROC'); plt.tight_layout(); plt.savefig(FIGURES_DIR / 'erosion_pr_roc_fused_active_bank_zone_v1.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    summary = {'checkpoint': str(ckpt_path), 'history_csv': str(history_path), 'metrics_csv': str(metrics_path), 'duration_sec': time.time() - start, **{k: float(v) for k,v in test_metrics.items()}}
    (REPORT_DIR / 'erosion_fused_active_bank_zone_v1_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print(json.dumps(summary, indent=2))
