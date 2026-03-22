from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio import features
from rasterio.features import sieve
from rasterio.windows import Window
from shapely.geometry import box, shape

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "outputs" / "report"
FIGURES = ROOT / "outputs" / "figures"

HAZARD_CATALOG_PATH = CONFIG / "hazard_catalog.json"
STUDY_AREAS_PATH = CONFIG / "study_areas_years.json"
README_PATH = ROOT / "README.md"
PATCH_DIR = PROCESSED / "patches_multihazard_ready"
PATCH_MANIFEST_SRC = PROCESSED / "patches_full256" / "patch_manifest.csv"
MULTI_PATCH_MANIFEST = REPORT / "multi_hazard_patch_manifest.csv"

EROSION_LABEL_DIR = RAW / "hazards" / "erosion" / "labels"
EROSION_AUX_DIR = RAW / "hazards" / "erosion" / "auxiliary"
LANDSLIDE_LABEL_DIR = RAW / "hazards" / "landslide" / "labels"
LANDSLIDE_AUX_DIR = RAW / "hazards" / "landslide" / "auxiliary"
FLOOD_MASK_SRC = PROCESSED / "masks" / "mask_unosat_20240619.tif"
FLOOD_MASK_DST = PROCESSED / "masks_multihazard" / "flood" / "flood_mask_sylhet_20240619.tif"
EROSION_MASK_DST = PROCESSED / "masks_multihazard" / "erosion" / "erosion_seed_mask_sylhet_20240619.tif"
LANDSLIDE_MASK_DST = PROCESSED / "masks_multihazard" / "landslide" / "landslide_seed_mask_sylhet_20240619.tif"
VV_PATH = PROCESSED / "sar" / "S1_VV_20240619.tif"
SLOPE_PATH = PROCESSED / "terrain" / "slope_sylhet.tif"
HAND_PATH = PROCESSED / "terrain" / "hand_sylhet.tif"
TWI_PATH = PROCESSED / "terrain" / "twi_sylhet.tif"
HYDRORIVERS_PATH = EROSION_AUX_DIR / "as_riv_15s" / "as_riv_15s.shp"


def ensure_dirs() -> None:
    for path in [EROSION_LABEL_DIR, LANDSLIDE_LABEL_DIR, FIGURES, REPORT, FLOOD_MASK_DST.parent, EROSION_MASK_DST.parent, LANDSLIDE_MASK_DST.parent]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_region_boxes() -> gpd.GeoDataFrame:
    study = load_json(STUDY_AREAS_PATH)
    rows = []
    for region in study["regions"]:
        minx, miny, maxx, maxy = region["bbox"]
        rows.append({
            "region_id": region["id"],
            "region_name": region["name"],
            "geometry": box(minx, miny, maxx, maxy),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def copy_flood_mask() -> dict:
    shutil.copy2(FLOOD_MASK_SRC, FLOOD_MASK_DST)
    with rasterio.open(FLOOD_MASK_DST) as src:
        arr = src.read(1)
    return {"positive_px": int(arr.sum()), "path": str(FLOOD_MASK_DST)}


def build_erosion_assets(region_boxes: gpd.GeoDataFrame) -> dict:
    rivers = gpd.read_file(HYDRORIVERS_PATH, bbox=tuple(region_boxes.total_bounds))
    rivers = rivers.to_crs("EPSG:4326")
    study_lines = gpd.overlay(rivers, region_boxes, how="intersection")
    study_lines_path = EROSION_LABEL_DIR / "hydrorivers_study_areas_lines.geojson"
    study_lines.to_file(study_lines_path, driver="GeoJSON")

    with rasterio.open(VV_PATH) as vv:
        vv_meta = vv.meta.copy()
        vv_bounds = vv.bounds
        vv_crs = vv.crs
        transformer = Transformer.from_crs(vv_crs, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(vv_bounds.left, vv_bounds.bottom)
        maxx, maxy = transformer.transform(vv_bounds.right, vv_bounds.top)
        sylhet_bbox = box(minx, miny, maxx, maxy)
        sylhet_rivers = rivers[rivers.intersects(sylhet_bbox)].copy()
        if sylhet_rivers.empty:
            raise RuntimeError("No HydroRIVERS lines intersect the Sylhet SAR footprint")
        sylhet_rivers = sylhet_rivers.to_crs(vv_crs)
        sylhet_rivers = sylhet_rivers[sylhet_rivers["UP_CELLS"] >= 500].copy()
        if sylhet_rivers.empty:
            sylhet_rivers = rivers[rivers.intersects(sylhet_bbox)].to_crs(vv_crs).copy()
        sylhet_rivers["buffer_m"] = np.select(
            [sylhet_rivers["UP_CELLS"] >= 50000, sylhet_rivers["UP_CELLS"] >= 5000, sylhet_rivers["UP_CELLS"] >= 500],
            [250, 150, 80],
            default=40,
        )
        sylhet_corridors = sylhet_rivers.copy()
        sylhet_corridors["geometry"] = [geom.buffer(float(buf)) for geom, buf in zip(sylhet_rivers.geometry, sylhet_rivers["buffer_m"])]

        sylhet_lines_path = EROSION_LABEL_DIR / "hydrorivers_sylhet_lines.geojson"
        sylhet_corridors_path = EROSION_LABEL_DIR / "erosion_seed_corridors_sylhet.geojson"
        sylhet_rivers.to_crs("EPSG:4326").to_file(sylhet_lines_path, driver="GeoJSON")
        sylhet_corridors.to_crs("EPSG:4326").to_file(sylhet_corridors_path, driver="GeoJSON")

        shapes = [(geom, 1) for geom in sylhet_corridors.geometry if geom and not geom.is_empty]
        erosion_mask = features.rasterize(
            shapes,
            out_shape=(vv.height, vv.width),
            transform=vv.transform,
            fill=0,
            dtype="uint8",
        )
        vv_meta.update(dtype="uint8", count=1, compress="lzw")
        with rasterio.open(EROSION_MASK_DST, "w", **vv_meta) as dst:
            dst.write(erosion_mask, 1)

    return {
        "study_lines_path": str(study_lines_path),
        "sylhet_lines_path": str(sylhet_lines_path),
        "sylhet_corridors_path": str(sylhet_corridors_path),
        "mask_path": str(EROSION_MASK_DST),
        "study_line_count": int(len(study_lines)),
        "sylhet_line_count": int(len(sylhet_rivers)),
        "positive_px": int(erosion_mask.sum()),
    }


def build_landslide_assets() -> dict:
    with rasterio.open(SLOPE_PATH) as slope_src, rasterio.open(HAND_PATH) as hand_src, rasterio.open(TWI_PATH) as twi_src:
        slope = slope_src.read(1, masked=True).filled(np.nan)
        hand = hand_src.read(1, masked=True).filled(np.nan)
        twi = twi_src.read(1, masked=True).filled(np.nan)
        meta = slope_src.meta.copy()
        transform = slope_src.transform
        crs = slope_src.crs

    valid = np.isfinite(slope) & np.isfinite(hand) & np.isfinite(twi)
    slope_q90 = float(np.nanquantile(slope[valid], 0.90))
    slope_q95 = float(np.nanquantile(slope[valid], 0.95))
    hand_q75 = float(np.nanquantile(hand[valid], 0.75))
    twi_q85 = float(np.nanquantile(twi[valid], 0.85))

    steep = slope >= slope_q95
    steep_wet_margin = (slope >= slope_q90) & (hand >= hand_q75) & (twi <= twi_q85)
    landslide_seed = (steep | steep_wet_margin) & valid
    landslide_seed = sieve(landslide_seed.astype("uint8"), size=128, connectivity=8)

    meta.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(LANDSLIDE_MASK_DST, "w", **meta) as dst:
        dst.write(landslide_seed.astype("uint8"), 1)

    polygons = []
    for geom, value in features.shapes(landslide_seed.astype("uint8"), mask=landslide_seed.astype(bool), transform=transform):
        if value == 1:
            polygons.append(shape(geom))
    landslide_polys = gpd.GeoDataFrame({"seed_id": np.arange(len(polygons), dtype=int)}, geometry=polygons, crs=crs)
    if not landslide_polys.empty:
        landslide_polys["area_m2"] = landslide_polys.area
    landslide_polys_path = LANDSLIDE_LABEL_DIR / "landslide_seed_areas_sylhet.geojson"
    landslide_polys.to_crs("EPSG:4326").to_file(landslide_polys_path, driver="GeoJSON")

    thresholds_path = LANDSLIDE_LABEL_DIR / "landslide_seed_thresholds.json"
    thresholds_path.write_text(json.dumps({
        "slope_q90": slope_q90,
        "slope_q95": slope_q95,
        "hand_q75": hand_q75,
        "twi_q85": twi_q85,
        "rule": "(slope>=q95) OR ((slope>=q90) AND (hand>=q75) AND (twi<=q85)), then sieve(size=128)",
    }, indent=2), encoding="utf-8")

    return {
        "polygons_path": str(landslide_polys_path),
        "thresholds_path": str(thresholds_path),
        "mask_path": str(LANDSLIDE_MASK_DST),
        "polygon_count": int(len(landslide_polys)),
        "positive_px": int(landslide_seed.sum()),
    }


def update_patch_targets() -> pd.DataFrame:
    patch_manifest = pd.read_csv(PATCH_MANIFEST_SRC)
    rows = []
    with rasterio.open(FLOOD_MASK_DST) as flood_src, rasterio.open(EROSION_MASK_DST) as erosion_src, rasterio.open(LANDSLIDE_MASK_DST) as landslide_src:
        for record in patch_manifest.itertuples(index=False):
            patch_path = PATCH_DIR / Path(record.path).name
            with np.load(patch_path) as npz:
                payload = {key: npz[key] for key in npz.files if key not in {"hazard_targets", "hazard_names"}}
                hazard_targets = npz["hazard_targets"].astype(np.float32)
                hazard_names = [str(x) for x in npz["hazard_names"].tolist()]
            order = {name: idx for idx, name in enumerate(hazard_names)}
            win = Window(int(record.col), int(record.row), 256, 256)
            flood_patch = flood_src.read(1, window=win).astype(np.float32)
            erosion_patch = erosion_src.read(1, window=win).astype(np.float32)
            landslide_patch = landslide_src.read(1, window=win).astype(np.float32)
            hazard_targets[order["flood"]] = flood_patch
            hazard_targets[order["erosion"]] = erosion_patch
            hazard_targets[order["landslide"]] = landslide_patch
            np.savez_compressed(patch_path, **payload, hazard_targets=hazard_targets, hazard_names=np.array(hazard_names, dtype="U32"))
            rows.append({
                "path": str(patch_path),
                "row": int(record.row),
                "col": int(record.col),
                "hazard_order": ",".join(hazard_names),
                "flood_positive_px": float(flood_patch.sum()),
                "erosion_positive_px": float(erosion_patch.sum()),
                "landslide_positive_px": float(landslide_patch.sum()),
                "has_flood": bool(flood_patch.sum() > 0),
                "has_erosion": bool(erosion_patch.sum() > 0),
                "has_landslide": bool(landslide_patch.sum() > 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(MULTI_PATCH_MANIFEST, index=False)
    return df


def build_reports(flood_info: dict, erosion_info: dict, landslide_info: dict, patch_df: pd.DataFrame) -> None:
    label_inventory = pd.DataFrame([
        {
            "hazard": "flood",
            "label_type": "validated_event_mask",
            "source": "UNOSAT / HDX",
            "path": flood_info["path"],
            "records_or_polygons": 1,
            "positive_pixels": flood_info["positive_px"],
            "note": "Validated Sylhet June 2024 flood extent aligned to SAR grid.",
        },
        {
            "hazard": "erosion",
            "label_type": "hydrorivers_seed_corridor",
            "source": "HydroSHEDS HydroRIVERS",
            "path": erosion_info["sylhet_corridors_path"],
            "records_or_polygons": erosion_info["sylhet_line_count"],
            "positive_pixels": erosion_info["positive_px"],
            "note": "Seed supervision from buffered river corridors over study areas and aligned Sylhet mask.",
        },
        {
            "hazard": "landslide",
            "label_type": "terrain_seed_polygon",
            "source": "Derived from aligned slope, HAND, and TWI",
            "path": landslide_info["polygons_path"],
            "records_or_polygons": landslide_info["polygon_count"],
            "positive_pixels": landslide_info["positive_px"],
            "note": "Seed supervision for steep, elevated, lower-wetness terrain within the aligned Sylhet stack.",
        },
    ])
    label_inventory.to_csv(REPORT / "multi_hazard_label_inventory.csv", index=False)

    dataset_manifest = pd.DataFrame([
        {"hazard": "flood", "status": "benchmark_validated", "supervision": "validated_event_mask", "asset_path": flood_info["path"], "positive_pixels": flood_info["positive_px"], "positive_patches": int(patch_df["has_flood"].sum())},
        {"hazard": "erosion", "status": "seed_supervision_ready", "supervision": "hydrorivers_bank_corridor_seed", "asset_path": erosion_info["mask_path"], "positive_pixels": erosion_info["positive_px"], "positive_patches": int(patch_df["has_erosion"].sum())},
        {"hazard": "landslide", "status": "seed_supervision_ready", "supervision": "terrain_based_seed_mask", "asset_path": landslide_info["mask_path"], "positive_pixels": landslide_info["positive_px"], "positive_patches": int(patch_df["has_landslide"].sum())},
    ])
    dataset_manifest.to_csv(REPORT / "multi_hazard_dataset_manifest.csv", index=False)

    training_plan = pd.DataFrame([
        {"hazard": "flood", "hazard_weight": 1.0, "task": "binary_segmentation", "validated_now": True, "supervision_mode": "validated_event_mask"},
        {"hazard": "erosion", "hazard_weight": 0.9, "task": "binary_segmentation", "validated_now": False, "supervision_mode": "hydrorivers_bank_corridor_seed"},
        {"hazard": "landslide", "hazard_weight": 0.8, "task": "binary_segmentation", "validated_now": False, "supervision_mode": "terrain_based_seed_mask"},
    ])
    training_plan.to_csv(REPORT / "multi_hazard_training_plan.csv", index=False)

    summary = f"""# Multi-Hazard Summary\n\nThe project now ships with on-disk supervision assets for all three hazards.\n\n- Flood remains the fully benchmarked branch with validated UNOSAT-aligned masks and saved CASA-Net metrics.\n- Erosion now includes HydroRIVERS-derived bank-corridor seed labels across the study areas and an aligned Sylhet seed mask.\n- Landslide now includes terrain-derived seed polygons and an aligned Sylhet seed mask based on slope, HAND, and TWI thresholds.\n\nCurrent positive patch counts from `patches_multihazard_ready`:\n\n- Flood: {int(patch_df['has_flood'].sum())}\n- Erosion: {int(patch_df['has_erosion'].sum())}\n- Landslide: {int(patch_df['has_landslide'].sum())}\n\nThis means the multi-hazard heads no longer depend on empty label folders. The flood branch is still the only branch with published benchmark metrics in this repository, while erosion and landslide now have initial trainable supervision assets and aligned masks.\n"""
    (REPORT / "multi_hazard_summary.md").write_text(summary, encoding="utf-8")


def update_hazard_catalog() -> None:
    catalog = load_json(HAZARD_CATALOG_PATH)
    catalog["notes"] = [
        "Flood branch has the fully validated CASA-Net benchmark results.",
        "Riverbank erosion now includes HydroRIVERS-derived bank-corridor seed labels and an aligned Sylhet seed mask.",
        "Landslide now includes terrain-derived seed polygons and an aligned Sylhet seed mask based on slope, HAND, and TWI.",
        "Combined risk layers are built from hazard probabilities, exposure, and vulnerability weights.",
    ]
    for hazard in catalog.get("hazards", []):
        if hazard["id"] == "flood":
            hazard["current_supervision"] = "validated_unosat_mask"
        elif hazard["id"] == "erosion":
            hazard["current_supervision"] = "hydrorivers_bank_corridor_seed"
            hazard["label_mode"] = "corridor_seed_to_raster"
        elif hazard["id"] == "landslide":
            hazard["current_supervision"] = "terrain_seed_mask"
            hazard["label_mode"] = "terrain_seed_to_raster"
    HAZARD_CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def update_readme() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    old_intro = "The flood branch remains the **fully validated benchmark branch** with trained CASA-Net results and complete figures. The erosion and landslide branches are now built into the data model, notebook workflow, patch format, and reporting structure so they can be activated as hazard-specific labels are added."
    new_intro = "The flood branch remains the **fully validated benchmark branch** with trained CASA-Net results and complete figures. The erosion and landslide branches now also ship with on-disk supervision assets: HydroRIVERS-derived bank-corridor seed labels for erosion and terrain-derived seed masks for landslide, so the multi-hazard heads no longer depend on empty label folders."
    text = text.replace(old_intro, new_intro)
    old_tail = "- Erosion and landslide branches are scaffolded and become trainable as labels are added to `data/raw/hazards/<hazard>/labels/`"
    new_tail = "- Erosion and landslide branches now include initial seed supervision assets under `data/raw/hazards/<hazard>/labels/` and aligned masks under `data/processed/masks_multihazard/`"
    text = text.replace(old_tail, new_tail)
    README_PATH.write_text(text, encoding="utf-8")


def update_notebook_training_text() -> None:
    nb_path = ROOT / "notebooks" / "05_casa_net_training.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    target = "This section extends CASA-Net into a **shared-backbone, multi-head model**. The current flood branch remains the validated benchmark branch, while erosion and landslide heads are trained whenever aligned labels exist in the multi-hazard patch set.\n"
    repl = "This section extends CASA-Net into a **shared-backbone, multi-head model**. Flood remains the validated benchmark branch, while erosion and landslide now use the initial seed supervision assets bundled in the multi-hazard patch set.\n"
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", [])
            joined = "".join(src)
            if target in joined:
                joined = joined.replace(target, repl)
                cell["source"] = [joined]
                changed = True
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def build_figures(patch_df: pd.DataFrame) -> None:
    metrics = pd.DataFrame([
        {"hazard": "flood", "positive_patches": int(patch_df['has_flood'].sum()), "positive_pixels": float(patch_df['flood_positive_px'].sum())},
        {"hazard": "erosion", "positive_patches": int(patch_df['has_erosion'].sum()), "positive_pixels": float(patch_df['erosion_positive_px'].sum())},
        {"hazard": "landslide", "positive_patches": int(patch_df['has_landslide'].sum()), "positive_pixels": float(patch_df['landslide_positive_px'].sum())},
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(metrics['hazard'], metrics['positive_patches'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Positive patches by hazard')
    axes[0].set_ylabel('Patch count')
    axes[1].bar(metrics['hazard'], metrics['positive_pixels'] / 1e6, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Positive pixels by hazard')
    axes[1].set_ylabel('Million pixels')
    fig.suptitle('Multi-hazard supervision coverage')
    fig.tight_layout()
    fig.savefig(FIGURES / 'multi_hazard_label_coverage.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    with rasterio.open(VV_PATH) as vv, rasterio.open(FLOOD_MASK_DST) as flood, rasterio.open(EROSION_MASK_DST) as erosion, rasterio.open(LANDSLIDE_MASK_DST) as landslide:
        vv_arr = vv.read(1)
        flood_arr = flood.read(1)
        erosion_arr = erosion.read(1)
        landslide_arr = landslide.read(1)
    vv_db = 10 * np.log10(np.clip(vv_arr, 1e-6, None))
    vmin, vmax = np.nanpercentile(vv_db, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mask, title, cmap in [
        (axes[0], flood_arr, 'Flood mask (validated)', 'Blues'),
        (axes[1], erosion_arr, 'Erosion seed mask', 'Oranges'),
        (axes[2], landslide_arr, 'Landslide seed mask', 'Greens'),
    ]:
        ax.imshow(vv_db, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, alpha=0.55)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Aligned multi-hazard masks over Sylhet Sentinel-1 VV')
    fig.tight_layout()
    fig.savefig(FIGURES / 'multi_hazard_seed_mask_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def refresh_diagram_manifest() -> None:
    manifest_path = FIGURES / 'evaluation_diagrams_manifest.csv'
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        df = pd.DataFrame(columns=['filename', 'category', 'source_notebook', 'description'])
    extras = pd.DataFrame([
        {"filename": "multi_hazard_label_coverage.png", "category": "dataset/evaluation", "source_notebook": "03_mask_preparation / 04_patch_generation", "description": "Positive patch and pixel coverage for flood, erosion, and landslide supervision assets."},
        {"filename": "multi_hazard_seed_mask_preview.png", "category": "dataset/label_preview", "source_notebook": "03_mask_preparation", "description": "Validated flood mask and initial erosion/landslide seed masks aligned to the Sylhet Sentinel-1 grid."},
    ])
    df = pd.concat([df, extras], ignore_index=True)
    df = df.drop_duplicates(subset=['filename'], keep='last').sort_values('filename')
    df.to_csv(manifest_path, index=False)


def main() -> None:
    ensure_dirs()
    region_boxes = build_region_boxes()
    flood_info = copy_flood_mask()
    erosion_info = build_erosion_assets(region_boxes)
    landslide_info = build_landslide_assets()
    patch_df = update_patch_targets()
    build_reports(flood_info, erosion_info, landslide_info, patch_df)
    update_hazard_catalog()
    update_readme()
    update_notebook_training_text()
    build_figures(patch_df)
    refresh_diagram_manifest()
    print(json.dumps({
        "flood_positive_patches": int(patch_df['has_flood'].sum()),
        "erosion_positive_patches": int(patch_df['has_erosion'].sum()),
        "landslide_positive_patches": int(patch_df['has_landslide'].sum()),
        "erosion_positive_pixels": int(patch_df['erosion_positive_px'].sum()),
        "landslide_positive_pixels": int(patch_df['landslide_positive_px'].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
