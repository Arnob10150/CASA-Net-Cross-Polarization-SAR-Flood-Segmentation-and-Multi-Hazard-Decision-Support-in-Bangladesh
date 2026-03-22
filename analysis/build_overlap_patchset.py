import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
PATCH_DIR = ROOT / "data" / "processed" / "patches_overlap128_optimistic"

VV_PATH = ROOT / "data" / "processed" / "sar" / "S1_VV_20240619.tif"
VH_PATH = ROOT / "data" / "processed" / "sar" / "S1_VH_20240619.tif"
MASK_PATH = ROOT / "data" / "processed" / "masks" / "mask_unosat_20240619.tif"
TERRAIN_PATHS = {
    "slope": ROOT / "data" / "processed" / "terrain" / "slope_sylhet.tif",
    "twi": ROOT / "data" / "processed" / "terrain" / "twi_sylhet.tif",
    "jrc": ROOT / "data" / "processed" / "terrain" / "jrc_water_sylhet.tif",
    "hand": ROOT / "data" / "processed" / "terrain" / "hand_sylhet.tif",
}

PATCH_SIZE = 256
STRIDE = 128
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def sliding_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def flood_bin(flood_pct):
    if flood_pct == 0:
        return "dry"
    if flood_pct < 0.01:
        return "very_low"
    if flood_pct < 0.05:
        return "low"
    if flood_pct < 0.20:
        return "medium"
    return "high"


def valid_fraction(arr):
    finite = np.isfinite(arr)
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def robust_terrain(arr):
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def read_patch(ds, row, col):
    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
    return ds.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)


def split_paths(manifest_df):
    stratify = manifest_df["bin"]
    if stratify.value_counts().min() < 3:
        stratify = None
    train_df, temp_df = train_test_split(
        manifest_df,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=SEED,
        shuffle=True,
        stratify=stratify,
    )
    stratify_temp = temp_df["bin"]
    if stratify_temp.value_counts().min() < 2:
        stratify_temp = None
    val_rel = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel),
        random_state=SEED,
        shuffle=True,
        stratify=stratify_temp,
    )
    return {
        "train": train_df["path"].tolist(),
        "val": val_df["path"].tolist(),
        "test": test_df["path"].tolist(),
    }


def compute_norm_stats(paths):
    vv_sum = 0.0
    vv_sq = 0.0
    vv_n = 0
    vh_sum = 0.0
    vh_sq = 0.0
    vh_n = 0
    flood_sum = 0.0
    flood_n = 0
    for path_str in paths:
        with np.load(path_str) as npz:
            vv = npz["vv"].astype(np.float32)
            vh = npz["vh"].astype(np.float32)
            mask = npz["mask"].astype(np.float32)
        vv_sum += float(vv.sum())
        vv_sq += float((vv ** 2).sum())
        vv_n += vv.size
        vh_sum += float(vh.sum())
        vh_sq += float((vh ** 2).sum())
        vh_n += vh.size
        flood_sum += float(mask.sum())
        flood_n += mask.size
    vv_mean = vv_sum / max(vv_n, 1)
    vh_mean = vh_sum / max(vh_n, 1)
    vv_std = max(((vv_sq / max(vv_n, 1)) - vv_mean ** 2) ** 0.5, 1e-6)
    vh_std = max(((vh_sq / max(vh_n, 1)) - vh_mean ** 2) ** 0.5, 1e-6)
    return {
        "vv_mean": vv_mean,
        "vv_std": vv_std,
        "vh_mean": vh_mean,
        "vh_std": vh_std,
        "train_flood_ratio": flood_sum / max(flood_n, 1),
    }


def main():
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(VV_PATH) as vv_ds, rasterio.open(VH_PATH) as vh_ds, rasterio.open(MASK_PATH) as mask_ds:
        terrain_ds = {name: rasterio.open(path) for name, path in TERRAIN_PATHS.items()}
        rows = sliding_positions(vv_ds.height, PATCH_SIZE, STRIDE)
        cols = sliding_positions(vv_ds.width, PATCH_SIZE, STRIDE)
        manifest_rows = []
        patch_idx = 0
        for row in rows:
            for col in cols:
                vv_patch = read_patch(vv_ds, row, col)
                vh_patch = read_patch(vh_ds, row, col)
                mask_patch = read_patch(mask_ds, row, col)
                terrain_stack = [read_patch(terrain_ds[name], row, col) for name in ["slope", "twi", "jrc", "hand"]]
                terrain_patch = np.stack([robust_terrain(x) for x in terrain_stack], axis=0)
                if min(valid_fraction(vv_patch), valid_fraction(vh_patch)) < 0.90:
                    continue
                vv_patch = np.nan_to_num(vv_patch, nan=0.0).astype(np.float32)[None, ...]
                vh_patch = np.nan_to_num(vh_patch, nan=0.0).astype(np.float32)[None, ...]
                mask_patch = np.nan_to_num(mask_patch, nan=0.0).astype(np.float32)
                mask_patch = (mask_patch > 0.5).astype(np.float32)[None, ...]
                patch_path = PATCH_DIR / f"patch_{patch_idx:04d}.npz"
                np.savez_compressed(patch_path, vv=vv_patch, vh=vh_patch, terrain=terrain_patch.astype(np.float32), mask=mask_patch)
                flood_pct = float(mask_patch.mean())
                manifest_rows.append({"path": str(patch_path), "row": int(row), "col": int(col), "flood_pct": flood_pct, "bin": flood_bin(flood_pct)})
                patch_idx += 1
        for ds in terrain_ds.values():
            ds.close()

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        raise RuntimeError("No overlap patches were generated")
    manifest_df.to_csv(PATCH_DIR / "patch_manifest.csv", index=False)

    split_index = split_paths(manifest_df)
    (PATCH_DIR / "split_index.json").write_text(json.dumps(split_index, indent=2))

    norm_stats = compute_norm_stats(split_index["train"])
    (PATCH_DIR / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))

    dataset_info = {
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "split_type": "optimistic_random_patch_split_with_overlap",
        "total_patches": int(len(manifest_df)),
        "train_count": int(len(split_index["train"])),
        "val_count": int(len(split_index["val"])),
        "test_count": int(len(split_index["test"])),
        "bins": manifest_df["bin"].value_counts().to_dict(),
        "sources": {
            "vv": str(VV_PATH),
            "vh": str(VH_PATH),
            "mask": str(MASK_PATH),
            "terrain": {k: str(v) for k, v in TERRAIN_PATHS.items()},
        },
    }
    (PATCH_DIR / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    print(json.dumps(dataset_info, indent=2))
    print(json.dumps(norm_stats, indent=2))


if __name__ == "__main__":
    main()
