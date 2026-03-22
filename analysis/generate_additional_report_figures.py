import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from shapely.geometry import box
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve, average_precision_score

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "report"
CONFIG = ROOT / "config" / "study_areas_years.json"
BORDER = ROOT / "data" / "raw" / "bangladesh.geojson"
SENTINEL = ROOT / "data" / "raw" / "sentinel1"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'font.size': 11,
})
sns.set_theme(style='whitegrid')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_country_boundary():
    gdf = gpd.read_file(BORDER).to_crs(4326)
    gdf['__country__'] = 1
    return gdf.dissolve(by='__country__')


def make_study_area_boxes():
    cfg = json.loads(CONFIG.read_text())
    regions = cfg['regions']
    country = load_country_boundary()
    bounds = country.total_bounds
    fig, ax = plt.subplots(figsize=(10, 8))
    country.plot(ax=ax, color='#f4f1ea', edgecolor='#444444', linewidth=0.8)
    country.boundary.plot(ax=ax, color='#222222', linewidth=1.0)
    palette = ['#d62828', '#1d3557', '#2a9d8f', '#e76f51', '#6a4c93', '#457b9d']
    for idx, region in enumerate(regions):
        minx, miny, maxx, maxy = region['bbox']
        geom = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='EPSG:4326')
        color = palette[idx % len(palette)]
        geom.boundary.plot(ax=ax, color=color, linewidth=2.2)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        ax.scatter([cx], [cy], s=18, color=color, zorder=5)
        ax.text(cx, maxy + 0.08, region['name'], ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.9))
    ax.set_xlim(bounds[0] - 0.25, bounds[2] + 0.25)
    ax.set_ylim(bounds[1] - 0.2, bounds[3] + 0.2)
    ax.set_title('Bangladesh study areas with validated bounding boxes and national outline')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    save(fig, 'study_area_bboxes.png')

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    country.plot(ax=ax2, color='#eef2f3', edgecolor='#222222', linewidth=0.7)
    for idx, region in enumerate(regions):
        minx, miny, maxx, maxy = region['bbox']
        color = palette[idx % len(palette)]
        rect = plt.Rectangle((minx, miny), maxx - minx, maxy - miny, fill=False, lw=2.2, ec=color)
        ax2.add_patch(rect)
        ax2.text(minx, maxy + 0.04, f"{region['name']}\n{minx:.1f},{miny:.1f} to {maxx:.1f},{maxy:.1f}",
                 color=color, fontsize=9, ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=color, alpha=0.9))
    ax2.set_xlim(bounds[0] - 0.25, bounds[2] + 0.25)
    ax2.set_ylim(bounds[1] - 0.2, bounds[3] + 0.2)
    ax2.set_title('Study-area bounding box reference map')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    save(fig2, 'study_area_bboxes_fixed.png')


def metric_panel(df, title, filename):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        order = df.sort_values(metric, ascending=False)
        colors = ['#d62828' if m == 'CASA-Net' else '#577590' for m in order['Model']]
        ax.bar(order['Model'], order[metric], color=colors)
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=20)
        for i, v in enumerate(order[metric]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    fig.suptitle(title, fontsize=16)
    save(fig, filename)


def heatmap(df, title, filename):
    order = df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'Dice']]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(order, annot=True, fmt='.3f', cmap='YlGnBu', cbar=True, ax=ax)
    ax.set_title(title)
    save(fig, filename)


def loss_score_curves():
    strict = pd.read_csv(REPORT / 'casanet_history_v3.csv')
    optimistic = pd.read_csv(REPORT / 'casanet_history_optimistic_tuned.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(strict['epoch'], strict['train_loss'], color='#d62828', marker='o')
    axes[0, 0].set_title('Strict run: training loss')
    axes[0, 1].plot(strict['epoch'], strict['val_f1'], label='Val F1', color='#2a9d8f', marker='o')
    axes[0, 1].plot(strict['epoch'], strict['val_iou'], label='Val IoU', color='#264653', marker='s')
    axes[0, 1].set_title('Strict run: validation performance')
    axes[0, 1].legend()
    axes[1, 0].plot(optimistic['epoch'], optimistic['train_loss'], color='#d62828', marker='o')
    axes[1, 0].set_title('Optimistic run: training loss')
    axes[1, 1].plot(optimistic['epoch'], optimistic['val_f1'], label='Val F1', color='#2a9d8f', marker='o')
    axes[1, 1].plot(optimistic['epoch'], optimistic['val_iou'], label='Val IoU', color='#264653', marker='s')
    axes[1, 1].set_title('Optimistic run: validation performance')
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
    fig.suptitle('CASA-Net loss and validation score curves')
    save(fig, 'casa_net_loss_score_curves.png')


def load_ns(script_path):
    source = Path(script_path).read_text()
    prefix = source.split("benchmark_variants = {")[0]
    ns = {}
    exec(prefix, ns)
    return ns


def predict_probs(ns, checkpoint_name, split='test', batch_size=4, dropout_p=0.1):
    Dataset = ns['SylhetDataset']
    DataLoader = ns['DataLoader']
    CASANet = ns['CASANet']
    DEVICE = ns['DEVICE']
    ds = Dataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = CASANet(dropout_p=dropout_p)
    state = torch.load(ROOT / 'models' / checkpoint_name, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for vv, vh, terrain, mask in loader:
            vv = vv.to(DEVICE, non_blocking=True)
            vh = vh.to(DEVICE, non_blocking=True)
            terrain = terrain.to(DEVICE, non_blocking=True)
            pred = model(vv, vh, terrain).detach().cpu().numpy()
            y_prob.append(pred.reshape(-1))
            y_true.append(mask.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def roc_pr_confusion_figures():
    strict_ns = load_ns(ROOT / 'analysis' / 'train_casanet_v3_rawterrain.py')
    opt_ns = load_ns(ROOT / 'analysis' / 'train_casanet_optimistic_tuned.py')
    y_true_s, y_prob_s = predict_probs(strict_ns, 'casa_net_best_v3.pth', batch_size=4, dropout_p=0.1)
    y_true_o, y_prob_o = predict_probs(opt_ns, 'casa_net_best_optimistic_tuned.pth', batch_size=6, dropout_p=0.0)

    strict_threshold = float(pd.read_csv(REPORT / 'test_metrics_casa_net_strict_precision_tuned.csv')['Threshold'].iloc[0])
    final_threshold = float(pd.read_csv(REPORT / 'test_metrics_casa_net_final.csv')['Threshold'].iloc[0])

    fpr_s, tpr_s, _ = roc_curve(y_true_s, y_prob_s)
    fpr_o, tpr_o, _ = roc_curve(y_true_o, y_prob_o)
    auc_s = auc(fpr_s, tpr_s)
    auc_o = auc(fpr_o, tpr_o)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_s, tpr_s, label=f'Strict CASA-Net (AUC={auc_s:.3f})', color='#264653', lw=2)
    ax.plot(fpr_o, tpr_o, label=f'Optimistic CASA-Net (AUC={auc_o:.3f})', color='#d62828', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC comparison for CASA-Net')
    ax.legend(loc='lower right')
    save(fig, 'roc_auc_casa_net_strict_vs_final.png')

    prec_s, rec_s, _ = precision_recall_curve(y_true_s, y_prob_s)
    prec_o, rec_o, _ = precision_recall_curve(y_true_o, y_prob_o)
    ap_s = average_precision_score(y_true_s, y_prob_s)
    ap_o = average_precision_score(y_true_o, y_prob_o)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_s, prec_s, label=f'Strict CASA-Net (AP={ap_s:.3f})', color='#264653', lw=2)
    ax.plot(rec_o, prec_o, label=f'Optimistic CASA-Net (AP={ap_o:.3f})', color='#d62828', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve comparison for CASA-Net')
    ax.legend(loc='lower left')
    save(fig, 'precision_recall_curve_casa_net_strict_vs_final.png')

    cm_s = confusion_matrix(y_true_s, (y_prob_s >= strict_threshold).astype(np.uint8))
    cm_o = confusion_matrix(y_true_o, (y_prob_o >= final_threshold).astype(np.uint8))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f'Strict confusion matrix\n(threshold={strict_threshold:.2f})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    sns.heatmap(cm_o, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1])
    axes[1].set_title(f'Optimistic confusion matrix\n(threshold={final_threshold:.2f})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    save(fig, 'confusion_matrix_casa_net_strict_vs_final.png')


def threshold_tradeoff():
    df = pd.read_csv(REPORT / 'casanet_strict_threshold_sweep.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['threshold'], df['test_precision'], label='Precision', color='#e76f51', lw=2)
    ax.plot(df['threshold'], df['test_recall'], label='Recall', color='#1d3557', lw=2)
    ax.plot(df['threshold'], df['test_f1'], label='F1', color='#2a9d8f', lw=2)
    ax.plot(df['threshold'], df['test_iou'], label='IoU', color='#264653', lw=2)
    ax.axvline(0.50, color='#d62828', linestyle='--', lw=1.5, label='Chosen strict threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Strict CASA-Net threshold tradeoff')
    ax.legend(ncol=2)
    save(fig, 'threshold_tradeoff_casa_net_strict.png')


def crop_quicklook(path):
    arr = np.asarray(Image.open(path).convert('RGB'))
    mask = arr.sum(axis=2) > 15
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return arr[y0:y1, x0:x1]


def regional_quicklooks():
    regions = [
        ('sunamganj', 'Sunamganj'),
        ('kurigram', 'Kurigram'),
        ('sirajganj', 'Sirajganj'),
    ]
    images = []
    for slug, label in regions:
        arr = crop_quicklook(SENTINEL / 'quicklooks_2025' / f'{slug}_2025.png')
        images.append((label, arr))
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.imshow(arr)
        ax.axis('off')
        ax.set_title(f'Sentinel-1 flood-season quicklook - {label} (2025)')
        save(fig, f'sentinel1_quicklook_{slug}_2025.png')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (label, arr) in zip(axes, images):
        ax.imshow(arr)
        ax.axis('off')
        ax.set_title(label)
    fig.suptitle('Regional Sentinel-1 quicklook comparison beyond Sylhet')
    save(fig, 'regional_sentinel1_quicklook_triptych.png')


def pick_shapefile(patterns):
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(patterns)


def multiyear_extent_examples():
    country = load_country_boundary()
    examples = [
        ('2016 Central Bangladesh', pick_shapefile([
            'data/raw/unosat_multi_year/central_bangladesh_2016/**/*ST20160724_Water_Extent.shp',
            'data/raw/unosat_multi_year/central_bangladesh_2016/**/*ST20160630_Water_Extent.shp',
        ])),
        ('2021 Chittagong Flooding', pick_shapefile([
            'data/raw/unosat_multi_year/chittagong_2021_08_08/**/*20210808_FloodExtent*.shp',
            'data/raw/unosat_multi_year/chittagong_2021_08_08/**/*20210729_FloodExtent*.shp',
        ])),
        ('2024 Sylhet Flooding', pick_shapefile([
            'data/raw/unosat/**/*S1_20240619_20240622_FloodExtent_Sylhet.shp',
        ])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, shp) in zip(axes, examples):
        gdf = gpd.read_file(shp).to_crs(4326)
        minx, miny, maxx, maxy = gdf.total_bounds
        country.boundary.plot(ax=ax, color='#888888', linewidth=0.4)
        gdf.plot(ax=ax, color='#2166ac', edgecolor='#0b3d91', linewidth=0.3, alpha=0.85)
        ax.set_xlim(minx - 0.3, maxx + 0.3)
        ax.set_ylim(miny - 0.2, maxy + 0.2)
        ax.set_title(title)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
    fig.suptitle('Multi-year UNOSAT flood-extent examples across Bangladesh')
    save(fig, 'unosat_multiyear_flood_extent_examples.png')


def main():
    strict_df = pd.read_csv(REPORT / 'benchmark_results_best_precision_tuned.csv')
    final_df = pd.read_csv(REPORT / 'benchmark_results_final.csv')
    make_study_area_boxes()
    metric_panel(strict_df, 'Strict benchmark metrics by model', 'metrics_overview_strict_precision_tuned.png')
    metric_panel(final_df, 'Optimistic final benchmark metrics by model', 'metrics_overview_final.png')
    heatmap(strict_df, 'Strict benchmark heatmap', 'model_comparison_heatmap_strict_precision_tuned.png')
    heatmap(final_df, 'Final benchmark heatmap', 'model_comparison_heatmap_final.png')
    loss_score_curves()
    roc_pr_confusion_figures()
    threshold_tradeoff()
    regional_quicklooks()
    multiyear_extent_examples()

    rows = []
    for path in sorted(FIGURES.glob('*')):
        if path.is_file():
            rows.append({
                'filename': path.name,
                'path': str(path),
                'size_bytes': path.stat().st_size,
                'modified': path.stat().st_mtime,
            })
    pd.DataFrame(rows).to_csv(FIGURES / 'evaluation_diagrams_manifest.csv', index=False)


if __name__ == '__main__':
    main()
