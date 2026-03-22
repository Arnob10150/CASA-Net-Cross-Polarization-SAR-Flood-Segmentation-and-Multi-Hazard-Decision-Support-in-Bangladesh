from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

ROOT = Path(r'f:\MAPATHON\sylhet_flood_2024')
PROCESSED = ROOT / 'data' / 'processed'
REPORT = ROOT / 'outputs' / 'report'
FIGURES = ROOT / 'outputs' / 'figures'

VV_PATH = PROCESSED / 'sar' / 'S1_VV_20240619.tif'
VH_PATH = PROCESSED / 'sar' / 'S1_VH_20240619.tif'
CORRIDOR_PATH = PROCESSED / 'masks_multihazard' / 'erosion' / 'erosion_seed_mask_sylhet_20240619.tif'
LANDSLIDE_BASE_PATH = PROCESSED / 'masks_multihazard' / 'landslide' / 'landslide_seed_mask_sylhet_20240619.tif'
SLOPE_PATH = PROCESSED / 'terrain' / 'slope_sylhet.tif'
EROSION_V2_PATH = PROCESSED / 'masks_multihazard' / 'erosion' / 'erosion_seed_mask_sylhet_20240619_v2.tif'
LANDSLIDE_V2_PATH = PROCESSED / 'masks_multihazard' / 'landslide' / 'landslide_seed_mask_sylhet_20240619_v2.tif'


def save_raster(out_path: Path, arr: np.ndarray, ref_path: Path) -> None:
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
    meta.update(dtype='uint8', count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr.astype('uint8'), 1)


def build_erosion_v2() -> dict:
    with rasterio.open(VV_PATH) as src:
        vv = src.read(1)
    with rasterio.open(CORRIDOR_PATH) as src:
        corridor = src.read(1).astype(bool)

    vvdb = 10 * np.log10(np.clip(vv, 1e-6, None))
    valid = (vv > 1e-6) & np.isfinite(vvdb) & corridor
    water_threshold_db = -2.0
    water = valid & (vvdb <= water_threshold_db)
    edge = ndi.binary_dilation(water, iterations=2) ^ ndi.binary_erosion(water, iterations=1)
    edge &= corridor

    save_raster(EROSION_V2_PATH, edge.astype('uint8'), VV_PATH)
    return {
        'mask_path': str(EROSION_V2_PATH),
        'threshold_db': water_threshold_db,
        'positive_pixels': int(edge.sum()),
        'corridor_pixels': int(corridor.sum()),
    }


def build_landslide_v2() -> dict:
    with rasterio.open(SLOPE_PATH) as src:
        slope = src.read(1).astype('float32')
    with rasterio.open(LANDSLIDE_BASE_PATH) as src:
        base = src.read(1).astype(bool)

    mean = ndi.uniform_filter(slope, size=5)
    mean_sq = ndi.uniform_filter(slope * slope, size=5)
    rough = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    rough_q80 = float(np.quantile(rough[np.isfinite(rough)], 0.80))
    slope_q95 = float(np.quantile(slope[np.isfinite(slope)], 0.95))

    mask = base & ((rough >= rough_q80) | (slope >= slope_q95))
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)
    mask = ndi.binary_fill_holes(mask)

    save_raster(LANDSLIDE_V2_PATH, mask.astype('uint8'), SLOPE_PATH)
    return {
        'mask_path': str(LANDSLIDE_V2_PATH),
        'rough_q80': rough_q80,
        'slope_q95': slope_q95,
        'positive_pixels': int(mask.sum()),
        'base_pixels': int(base.sum()),
    }


def build_preview() -> None:
    with rasterio.open(VV_PATH) as src:
        vv = src.read(1)
    vvdb = 10 * np.log10(np.clip(vv, 1e-6, None))
    vmin, vmax = np.nanpercentile(vvdb[vv > 1e-6], [2, 98])
    with rasterio.open(EROSION_V2_PATH) as src:
        erosion = src.read(1)
    with rasterio.open(LANDSLIDE_V2_PATH) as src:
        landslide = src.read(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].imshow(vvdb, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].imshow(np.ma.masked_where(erosion == 0, erosion), cmap='Oranges', alpha=0.55)
    axes[0].set_title('Erosion v2 mask over VV')
    axes[0].axis('off')
    axes[1].imshow(vvdb, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].imshow(np.ma.masked_where(landslide == 0, landslide), cmap='Greens', alpha=0.55)
    axes[1].set_title('Landslide v2 mask over VV')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES / 'multi_hazard_v2_mask_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    erosion_info = build_erosion_v2()
    landslide_info = build_landslide_v2()
    build_preview()
    out = {'erosion_v2': erosion_info, 'landslide_v2': landslide_info}
    (REPORT / 'hazard_v2_assets_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
