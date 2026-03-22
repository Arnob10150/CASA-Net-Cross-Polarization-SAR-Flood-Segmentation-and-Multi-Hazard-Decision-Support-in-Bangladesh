import os
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
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import segmentation_models_pytorch as smp

os.environ['TORCH_HOME'] = r'f:\MAPATHON\sylhet_flood_2024\.torch'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(r'f:\MAPATHON\sylhet_flood_2024')
PATCH_DIR = ROOT / 'data' / 'processed' / 'patches_overlap128_optimistic'
MODELS_DIR = ROOT / 'models'
FIGURES_DIR = ROOT / 'outputs' / 'figures'
REPORT_DIR = ROOT / 'outputs' / 'report'
for path in [MODELS_DIR, FIGURES_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('CUDA device not available')
print('Using device:', DEVICE, torch.cuda.get_device_name(0))

SPLIT_INDEX_PATH = PATCH_DIR / 'split_index.json'
NORM_STATS_PATH = PATCH_DIR / 'norm_stats.json'
MANIFEST_PATH = PATCH_DIR / 'patch_manifest.csv'
BATCH_SIZE = 4
EPOCHS = 14
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-5


def robust_norm(arr):
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def normalize(arr, mean, std):
    return (arr - mean) / max(std, 1e-6)


def normalize_terrain(arr):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = robust_norm(arr[i])
    return out


class SylhetDataset(Dataset):
    def __init__(self, split='train', augment=False):
        self.split = split
        self.augment = augment
        self.split_index = json.loads(SPLIT_INDEX_PATH.read_text())
        self.paths = [Path(p) for p in self.split_index[split]]
        self.norm = json.loads(NORM_STATS_PATH.read_text())
        self.manifest = pd.read_csv(MANIFEST_PATH)
        self.flood_map = dict(zip(self.manifest['path'], self.manifest['flood_pct']))
        self.transform = A.Compose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)],
            additional_targets={'vh': 'image', 't0': 'image', 't1': 'image', 't2': 'image', 't3': 'image', 'mask': 'mask'}
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        npz = np.load(self.paths[idx])
        vv = normalize(npz['vv'].astype(np.float32), self.norm['vv_mean'], self.norm['vv_std'])
        vh = normalize(npz['vh'].astype(np.float32), self.norm['vh_mean'], self.norm['vh_std'])
        terrain = npz['terrain'].astype(np.float32)
        mask = npz['mask'].astype(np.float32)
        if self.augment:
            aug = self.transform(image=vv[0], vh=vh[0], t0=terrain[0], t1=terrain[1], t2=terrain[2], t3=terrain[3], mask=mask[0])
            vv = aug['image'][None, ...] + np.random.normal(0, 0.03, size=vv.shape).astype(np.float32)
            vh = aug['vh'][None, ...] + np.random.normal(0, 0.03, size=vh.shape).astype(np.float32)
            terrain = np.stack([aug['t0'], aug['t1'], aug['t2'], aug['t3']], axis=0)
            mask = aug['mask'][None, ...]
        return torch.tensor(vv), torch.tensor(vh), torch.tensor(terrain), torch.tensor(mask)

    def sample_weights(self):
        weights = []
        for path in self.paths:
            flood_pct = float(self.flood_map[str(path)])
            if flood_pct == 0:
                w = 1.0
            elif flood_pct < 0.01:
                w = 2.0
            elif flood_pct < 0.05:
                w = 3.0
            elif flood_pct < 0.2:
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
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'IoU': float(intersection / max(union, 1)),
        'Dice': float((2 * intersection) / max((y_true == 1).sum() + (y_pred == 1).sum(), 1)),
    }


def sweep_thresholds(y_true, y_prob):
    rows = []
    for th in np.arange(0.2, 0.81, 0.05):
        rows.append({'threshold': float(th), **metrics_from_prob(y_true, y_prob, float(th))})
    df = pd.DataFrame(rows)
    best = df.sort_values(['IoU', 'F1', 'Dice'], ascending=False).iloc[0]
    return float(best['threshold']), df


train_ds = SylhetDataset('train', augment=False)
val_ds = SylhetDataset('val', augment=False)
test_ds = SylhetDataset('test', augment=False)
train_sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print('Dataset sizes:', len(train_ds), len(val_ds), len(test_ds))
print('Train flood ratio:', json.loads(NORM_STATS_PATH.read_text())['train_flood_ratio'])


class CPAG(nn.Module):
    def __init__(self, vv_channels, vh_channels, inter_channels, attn_size=8):
        super().__init__()
        self.q = nn.Conv2d(vv_channels, inter_channels, 1)
        self.k = nn.Conv2d(vh_channels, inter_channels, 1)
        self.v = nn.Conv2d(vh_channels, vh_channels, 1)
        self.proj = nn.Conv2d(vh_channels, vv_channels, 1)
        self.scale = inter_channels ** -0.5
        self.attn_size = attn_size

    def forward(self, f_vv, g_vh):
        if g_vh.shape[-2:] != f_vv.shape[-2:]:
            g_vh = F.interpolate(g_vh, size=f_vv.shape[-2:], mode='bilinear', align_corners=False)
        vv_small = F.adaptive_avg_pool2d(f_vv, self.attn_size)
        vh_small = F.adaptive_avg_pool2d(g_vh, self.attn_size)
        q = self.q(vv_small).flatten(2)
        k = self.k(vh_small).flatten(2)
        v = self.v(vh_small).flatten(2)
        b, _, hw = q.shape
        side = int(hw ** 0.5)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, -1, side, side)
        out = self.proj(out)
        out = F.interpolate(out, size=f_vv.shape[-2:], mode='bilinear', align_corners=False)
        return f_vv + out


class HaorTerrainBranch(nn.Module):
    def __init__(self, in_channels=4, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.generators = nn.ModuleList([nn.Linear(64, 2 * c) for c in decoder_channels])

    def forward(self, terrain):
        context = self.encoder(terrain).flatten(1)
        return [gen(context) for gen in self.generators]


def apply_film(x, params):
    gamma, beta = params.chunk(2, dim=1)
    gamma = 1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
    beta = 0.1 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
    return gamma * x + beta


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        backbone = torchvision.models.resnet34(weights=weights)
        old_weight = backbone.conv1.weight.detach().clone()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data.copy_(old_weight.mean(dim=1, keepdim=True))
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
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        old_weight = backbone.features[0][0].weight.detach().clone()
        backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        backbone.features[0][0].weight.data.copy_(old_weight.mean(dim=1, keepdim=True))
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
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(512, 256, 256),
            DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64),
        ])
        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, fused, film_params):
        x = self.blocks[0](fused[-1], fused[-2])
        x = apply_film(x, film_params[0])
        x = self.blocks[1](x, fused[-3])
        x = apply_film(x, film_params[1])
        x = self.blocks[2](x, fused[-4])
        x = apply_film(x, film_params[2])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.final(x)
        x = apply_film(x, film_params[3])
        return x


class UncertaintyHead(nn.Module):
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
        self.head = UncertaintyHead(32, dropout_p)

    def forward(self, x_vv, x_vh, x_terrain):
        size = x_vv.shape[-2:]
        vv_feats = self.vv_encoder(x_vv)
        vh_feats = self.vh_encoder(x_vh)
        fused = [self.cpag[i](vv_feats[i], vh_feats[i]) for i in range(4)]
        film = self.terrain_branch(x_terrain)
        x = self.decoder(fused, film)
        x = self.head(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class BaselineWrapper(nn.Module):
    def __init__(self, arch_name):
        super().__init__()
        builders = {
            'U-Net': lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1, activation='sigmoid'),
            'U-Net++': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1, activation='sigmoid'),
            'FPN': lambda: smp.FPN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1, activation='sigmoid'),
            'DeepLabV3+': lambda: smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1, activation='sigmoid'),
            'MAnet': lambda: smp.MAnet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=2, classes=1, activation='sigmoid'),
        }
        self.model = builders[arch_name]()

    def forward(self, x_vv, x_vh, x_terrain):
        return self.model(torch.cat([x_vv, x_vh], dim=1))


def collect_probs(model, loader):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(DEVICE, non_blocking=True)
            vh = vh.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            pred = model(vv, vh, terrain)
            probs.append(pred.cpu().numpy().ravel())
            targets.append(mask.numpy().ravel())
    return np.concatenate(targets).astype(np.uint8), np.concatenate(probs)


def evaluate_epoch(model, loader, criterion, threshold=0.5):
    model.eval()
    losses = []
    probs, targets = [], []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(DEVICE, non_blocking=True)
            vh = vh.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            pred = model(vv, vh, terrain)
            losses.append(float(criterion(pred, mask).item()))
            probs.append(pred.cpu().numpy().ravel())
            targets.append(mask.cpu().numpy().ravel())
    y_true = np.concatenate(targets).astype(np.uint8)
    y_prob = np.concatenate(probs)
    metrics = metrics_from_prob(y_true, y_prob, threshold)
    metrics['loss'] = float(np.mean(losses))
    return metrics


def train_variant(name, model, pos_weight):
    model = model.to(DEVICE)
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_state = deepcopy(model.state_dict())
    best_iou = -1.0
    best_threshold = 0.5
    wait = 0
    history = []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for vv, vh, terrain, mask in train_loader:
            vv = vv.to(DEVICE, non_blocking=True)
            vh = vh.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(vv, vh, terrain)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()
        val_true, val_prob = collect_probs(model, val_loader)
        th, th_df = sweep_thresholds(val_true, val_prob)
        val_metrics = metrics_from_prob(val_true, val_prob, th)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'val_threshold': th, 'val_iou': val_metrics['IoU'], 'val_f1': val_metrics['F1'], 'val_dice': val_metrics['Dice']})
        print(name, 'epoch', epoch, 'val_threshold', f'{th:.2f}', 'val_iou', f"{val_metrics['IoU']:.4f}", 'val_f1', f"{val_metrics['F1']:.4f}")
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            best_state = deepcopy(model.state_dict())
            best_threshold = th
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    return best_state, pd.DataFrame(history), time.time() - start, best_threshold


benchmark_variants = {
    'CASA-Net': CASANet(dropout_p=0.0),
}

train_ratio = json.loads(NORM_STATS_PATH.read_text())['train_flood_ratio']
pos_weight = max(1.0, (1 - train_ratio) / max(train_ratio, 1e-6))
print('Using pos_weight:', pos_weight)
rows = []
histories = {}
casa_true = casa_pred = casa_prob = None
execution = {'device': torch.cuda.get_device_name(0), 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'train_count': len(train_ds), 'val_count': len(val_ds), 'test_count': len(test_ds), 'patch_dir': str(PATCH_DIR), 'pos_weight': pos_weight}

for name, model in benchmark_variants.items():
    print('\n=== Training', name, '===')
    best_state, hist_df, duration, best_threshold = train_variant(name, model, pos_weight)
    histories[name] = hist_df
    hist_name = name.lower().replace('+', 'plus').replace('-', '').replace(' ', '_')
    hist_df.to_csv(REPORT_DIR / 'casanet_history_optimistic_tuned.csv', index=False)
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    ckpt_name = 'casa_net_best_optimistic_tuned.pth'
    torch.save(model.state_dict(), MODELS_DIR / ckpt_name)
    y_true, y_prob = collect_probs(model, test_loader)
    metrics = metrics_from_prob(y_true, y_prob, best_threshold)
    rows.append({'Model': name, **metrics, 'Threshold': best_threshold, 'Params': int(sum(p.numel() for p in model.parameters())), 'TrainSeconds': round(duration, 2)})
    if name == 'CASA-Net':
        casa_true = y_true
        casa_prob = y_prob
        casa_pred = (y_prob >= best_threshold).astype(np.uint8)
        pd.DataFrame([{**metrics, 'Threshold': best_threshold}]).to_csv(REPORT_DIR / 'test_metrics_casa_net_optimistic_tuned.csv', index=False)
    del model
    torch.cuda.empty_cache()

benchmark_df = pd.DataFrame(rows).sort_values(['IoU', 'F1', 'Dice'], ascending=False).reset_index(drop=True)
benchmark_df['Rank'] = np.arange(1, len(benchmark_df) + 1)
benchmark_df.to_csv(REPORT_DIR / 'benchmark_results_casanet_only_optimistic_tuned.csv', index=False)

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
pos = np.arange(len(benchmark_df))
ax.bar(pos - 0.25, benchmark_df['IoU'], width=0.25, label='IoU', color='#264653')
ax.bar(pos, benchmark_df['F1'], width=0.25, label='F1', color='#2a9d8f')
ax.bar(pos + 0.25, benchmark_df['Dice'], width=0.25, label='Dice', color='#e76f51')
ax.set_xticks(pos)
ax.set_xticklabels(benchmark_df['Model'], rotation=15)
ax.set_ylim(0, 1)
ax.set_title('Improved CUDA benchmark on full-scene 256 px patches')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'benchmark_model_comparison_casanet_only_optimistic_tuned.png', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
leaderboard = benchmark_df.sort_values('Rank')
colors = ['#d62828' if m == 'CASA-Net' else '#577590' for m in leaderboard['Model']]
ax.barh(leaderboard['Model'], leaderboard['IoU'], color=colors)
ax.invert_yaxis()
ax.set_xlabel('Test IoU')
ax.set_title('Improved benchmark leaderboard')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'benchmark_ranked_leaderboard_casanet_only_optimistic_tuned.png', dpi=300, bbox_inches='tight')
plt.close(fig)

if casa_true is not None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(confusion_matrix(casa_true, casa_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title('CASA-Net confusion matrix (improved run)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix_casa_net_optimistic_tuned.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    precision_vals, recall_vals, _ = precision_recall_curve(casa_true, casa_prob)
    fpr, tpr, _ = roc_curve(casa_true, casa_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    axes[0].plot(recall_vals, precision_vals, color='#1d3557', linewidth=2)
    axes[0].set_title('Precision-Recall')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[1].plot(fpr, tpr, color='#e76f51', linewidth=2, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].legend()
    axes[1].set_title('ROC')
    axes[1].set_xlabel('False positive rate')
    axes[1].set_ylabel('True positive rate')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'precision_recall_roc_casa_net_optimistic_tuned.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

execution['top_model'] = benchmark_df.iloc[0]['Model']
execution['top_iou'] = float(benchmark_df.iloc[0]['IoU'])
execution['top_f1'] = float(benchmark_df.iloc[0]['F1'])
(REPORT_DIR / 'training_execution_summary_casanet_optimistic_tuned.json').write_text(json.dumps(execution, indent=2))
print(benchmark_df.to_string(index=False))


