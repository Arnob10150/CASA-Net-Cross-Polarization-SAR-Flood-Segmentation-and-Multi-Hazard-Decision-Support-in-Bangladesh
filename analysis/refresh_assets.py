import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import requests
from shapely.geometry import box

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "report"
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "study_areas_years.json"
README_PATH = ROOT / "README.md"

VV_TIF = DATA_DIR / "raw" / "sentinel1" / "rtc_clipped" / "S1A_IW_GRDH_1SDV_20240619T234749_20240619T234814_054399_069E70_rtc_vv_sylhet_clip.tif"
VH_TIF = DATA_DIR / "raw" / "sentinel1" / "rtc_clipped" / "S1A_IW_GRDH_1SDV_20240619T234749_20240619T234814_054399_069E70_rtc_vh_sylhet_clip.tif"
UNOSAT_SHP = DATA_DIR / "raw" / "unosat" / "TC20240502BGD_SHP" / "TC20240502BGD_SHP" / "S1_20240619_20240622_FloodExtent_Sylhet.shp"
QUICKLOOK_2025_PATH = DATA_DIR / "raw" / "sentinel1" / "sample_quicklook_sylhet_2025.png"
QUICKLOOK_2025_FIG = FIG_DIR / "sample_sentinel1_quicklook_2025.png"
QUICKLOOK_2025_DIR = DATA_DIR / "raw" / "sentinel1" / "quicklooks_2025"

PC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"


def ensure_dirs():
    for path in [FIG_DIR, REPORT_DIR, QUICKLOOK_2025_PATH.parent, QUICKLOOK_2025_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def ensure_2025_in_config(config):
    years = sorted(set(config.get("years", [])) | {2025})
    config["years"] = years
    save_config(config)
    if README_PATH.exists():
        text = README_PATH.read_text(encoding="utf-8")
        text = text.replace("8 years (2017–2024)", "9 years (2017-2025)")
        text = text.replace("8 years (2017-2024)", "9 years (2017-2025)")
        text = text.replace("Years: 2017–2024", "Years: 2017-2025")
        text = text.replace("Years: 2017-2024", "Years: 2017-2025")
        README_PATH.write_text(text, encoding="utf-8")
    return config


def read_db_for_plot(path, max_dim=1600):
    with rasterio.open(path) as src:
        scale = max(src.height / max_dim, src.width / max_dim, 1)
        out_h = max(1, int(src.height / scale))
        out_w = max(1, int(src.width / scale))
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            masked=True,
            resampling=Resampling.average,
        ).astype("float32")
        arr = np.ma.masked_invalid(arr)
        db = 10.0 * np.log10(np.ma.clip(arr, 1e-6, None))
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs
    return db, extent, crs


def percentile_limits(arr, low=2, high=98):
    valid = arr.compressed()
    return float(np.percentile(valid, low)), float(np.percentile(valid, high))


def make_real_rtc_figure():
    vv_db, extent, _ = read_db_for_plot(VV_TIF)
    vh_db, _, _ = read_db_for_plot(VH_TIF)
    vv_vmin, vv_vmax = percentile_limits(vv_db)
    vh_vmin, vh_vmax = percentile_limits(vh_db)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im0 = axes[0].imshow(vv_db, cmap="gray", extent=extent, origin="upper", vmin=vv_vmin, vmax=vv_vmax)
    axes[0].set_title("Sentinel-1 RTC VV backscatter (dB) - Sylhet - 2024-06-19")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="VV dB")

    im1 = axes[1].imshow(vh_db, cmap="gray", extent=extent, origin="upper", vmin=vh_vmin, vmax=vh_vmax)
    axes[1].set_title("Sentinel-1 RTC VH backscatter (dB) - Sylhet - 2024-06-19")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="VH dB")

    out = FIG_DIR / "real_rtc_vv_vh_sylhet_june2024.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_real_overlay_figure():
    vv_db, extent, crs = read_db_for_plot(VV_TIF)
    vv_vmin, vv_vmax = percentile_limits(vv_db)
    gdf = gpd.read_file(UNOSAT_SHP).to_crs(crs)
    raster_box = box(extent[0], extent[2], extent[1], extent[3])
    gdf = gdf[gdf.intersects(raster_box)].copy()
    if not gdf.empty:
        gdf["geometry"] = gdf.geometry.intersection(raster_box)
        gdf = gdf[~gdf.geometry.is_empty]

    fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    ax.imshow(vv_db, cmap="gray", extent=extent, origin="upper", vmin=vv_vmin, vmax=vv_vmax)
    if not gdf.empty:
        gdf.plot(ax=ax, facecolor="#2b8cbe", edgecolor="#08519c", linewidth=0.8, alpha=0.35)
        gdf.boundary.plot(ax=ax, color="#08306b", linewidth=1.0)
    ax.set_title("UNOSAT flood extent over Sentinel-1 RTC VV - Sylhet - 2024-06-19 to 2024-06-22")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.text(
        0.02,
        0.03,
        "Blue polygon: UNOSAT June 19-22 flood extent",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"),
    )
    out = FIG_DIR / "real_unosat_overlay_sylhet_june2024.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def query_scene_coverage(config):
    import planetary_computer
    from pystac_client import Client

    client = Client.open(PC_API, modifier=planetary_computer.sign_inplace)
    rows = []
    sample_rows = []
    sylhet_2025_item = None

    flood_start = config["window_templates"]["pre_flood"][0]
    flood_end = config["window_templates"]["post_flood"][1]

    for region in config["regions"]:
        for year in config["years"]:
            search = client.search(
                collections=["sentinel-1-rtc"],
                bbox=region["bbox"],
                datetime=f"{year}-{flood_start}T00:00:00Z/{year}-{flood_end}T23:59:59Z",
                limit=500,
            )
            items = sorted(
                list(search.items()),
                key=lambda item: item.datetime or pd.Timestamp("1900-01-01", tz="UTC"),
            )
            rows.append(
                {
                    "region_id": region["id"],
                    "region_name": region["name"],
                    "year": year,
                    "scene_count": len(items),
                }
            )
            if year == 2025 and items:
                first_item = items[0]
                sample_rows.append(
                    {
                        "region_id": region["id"],
                        "region_name": region["name"],
                        "year": year,
                        "item_id": first_item.id,
                        "datetime": first_item.datetime.isoformat(),
                    }
                )
                preview_asset = None
                for key in ["rendered_preview", "thumbnail", "preview"]:
                    if key in first_item.assets:
                        preview_asset = first_item.assets[key]
                        break
                if preview_asset is not None:
                    region_preview_path = QUICKLOOK_2025_DIR / f"{region['id']}_2025.png"
                    try:
                        response = requests.get(preview_asset.href, timeout=120)
                        response.raise_for_status()
                        region_preview_path.write_bytes(response.content)
                    except Exception as exc:
                        print(f"Could not download {region['name']} 2025 quicklook: {exc}")
                if region["id"] == "sylhet" and sylhet_2025_item is None:
                    sylhet_2025_item = first_item

    manifest = pd.DataFrame(rows).sort_values(["region_name", "year"]).reset_index(drop=True)
    manifest.to_csv(REPORT_DIR / "scene_coverage_manifest.csv", index=False)

    if sample_rows:
        sample_df = pd.DataFrame(sample_rows).drop_duplicates(["region_id", "year"]).sort_values(["region_name", "year"])
        sample_df.to_csv(REPORT_DIR / "scene_samples_2025.csv", index=False)
    else:
        pd.DataFrame(columns=["region_id", "region_name", "year", "item_id", "datetime"]).to_csv(REPORT_DIR / "scene_samples_2025.csv", index=False)

    if sylhet_2025_item is not None:
        preview_asset = None
        for key in ["rendered_preview", "thumbnail", "preview"]:
            if key in sylhet_2025_item.assets:
                preview_asset = sylhet_2025_item.assets[key]
                break
        if preview_asset is not None:
            try:
                response = requests.get(preview_asset.href, timeout=120)
                response.raise_for_status()
                QUICKLOOK_2025_PATH.write_bytes(response.content)
                QUICKLOOK_2025_FIG.write_bytes(response.content)
            except Exception as exc:
                print(f"Could not download 2025 quicklook: {exc}")

    return manifest


def make_scene_coverage_figures(df):
    years = sorted(df["year"].unique())
    region_order = [name for name in df[["region_name"]].drop_duplicates()["region_name"]]
    pivot = df.pivot(index="region_name", columns="year", values="scene_count").reindex(index=region_order, columns=years)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlGnBu")
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_yticks(range(len(region_order)), region_order)
    ax.set_title("Sentinel-1 RTC scene coverage by region and year (2017-2025)")
    for i in range(len(region_order)):
        for j in range(len(years)):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Scene count")
    fig.savefig(FIG_DIR / "scene_count_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    by_year = df.groupby("year", as_index=False)["scene_count"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(by_year["year"].astype(str), by_year["scene_count"], color="#2b8cbe")
    ax.set_title("Total Sentinel-1 RTC scenes across study areas (2017-2025)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total scenes")
    fig.savefig(FIG_DIR / "scene_count_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    by_region = df.groupby("region_name", as_index=False)["scene_count"].sum().sort_values("scene_count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(by_region["region_name"], by_region["scene_count"], color="#7bccc4")
    ax.set_title("Total Sentinel-1 RTC scenes by study area (2017-2025)")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total scenes")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(FIG_DIR / "scene_count_by_region.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for region_name, group in df.groupby("region_name"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["scene_count"], marker="o", linewidth=2, label=region_name)
    ax.set_title("Scene coverage trend by region (2017-2025)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Scene count")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(FIG_DIR / "scene_count_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    availability = (pivot.values > 0).astype(int)
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    im = ax.imshow(availability, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_yticks(range(len(region_order)), region_order)
    ax.set_title("Study catalog availability timeline (2017-2025)")
    for i in range(len(region_order)):
        for j in range(len(years)):
            ax.text(j, i, "Yes" if availability[i, j] else "No", ha="center", va="center", fontsize=7, color="black")
    fig.savefig(FIG_DIR / "study_catalog_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    only_2025 = df[df["year"] == 2025].sort_values("scene_count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(only_2025["region_name"], only_2025["scene_count"], color="#fd8d3c")
    ax.set_title("Sentinel-1 RTC scenes found for flood-season 2025")
    ax.set_xlabel("Region")
    ax.set_ylabel("Scene count")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(FIG_DIR / "scene_count_2025_only.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dataset_description_figures(config, manifest_df):
    records = [
        ["Sentinel-1 RTC", "SAR VV/VH", "2017-2025", f"{len(config['regions'])} study areas", "Training/inference backbone"],
        ["UNOSAT flood extent", "Vector labels", "2024 local package", "Sylhet June 19-22", "Reference flood mask"],
        ["FFWC gauge data", "Time series", "2024 target + catalog support", "River stations", "Threshold calibration"],
        ["WorldPop", "Population raster", "2020 baseline", "Bangladesh", "Exposure estimation"],
        ["Infrastructure", "Points/polygons", "HDX sources", "Facilities + admin units", "Impact overlay"],
        ["SRTM/JRC/HAND", "Terrain layers", "Static", "AOI-aligned", "HTC terrain conditioning"],
    ]
    df = pd.DataFrame(records, columns=["Dataset", "Type", "Years", "Coverage", "Role in pipeline"])

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9edf7")
            cell.set_text_props(weight="bold")
    ax.set_title("Dataset description overview for the CASA-Net flood benchmark", fontsize=13, pad=14)
    fig.savefig(FIG_DIR / "dataset_description_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for region in config["regions"]:
        west, south, east, north = region["bbox"]
        rect = patches.Rectangle((west, south), east - west, north - south, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(west + 0.02, north + 0.02, region["name"], fontsize=9)
    ax.set_xlim(88.8, 92.8)
    ax.set_ylim(23.8, 26.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Study area bounding boxes across Bangladesh flood-prone regions")
    ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / "study_area_bboxes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sources = [
        "Sentinel-1 RTC",
        "UNOSAT",
        "DFO",
        "Copernicus EMS",
        "WorldPop",
        "HDX infrastructure",
        "FFWC gauges",
        "SRTM/JRC/HAND",
    ]
    tasks = [
        "SAR inputs",
        "Flood labels",
        "Exposure",
        "Thresholds",
        "Terrain context",
    ]
    matrix = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)), tasks, rotation=20, ha="right")
    ax.set_yticks(range(len(sources)), sources)
    ax.set_title("Data source-to-task matrix for the flood research pipeline")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "Yes" if matrix[i, j] else "-", ha="center", va="center", fontsize=8)
    fig.savefig(FIG_DIR / "data_source_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    shp_dir = DATA_DIR / "raw" / "unosat" / "TC20240502BGD_SHP" / "TC20240502BGD_SHP"
    categories = Counter()
    for shp in shp_dir.glob("*.shp"):
        name = shp.stem.lower()
        if "floodextent" in name or "floodwater" in name:
            categories["Flood extent / water"] += 1
        elif "analysisextent" in name:
            categories["Analysis extent"] += 1
        elif "damageassessment" in name:
            categories["Damage assessment"] += 1
        else:
            categories["Other"] += 1
    cat_df = pd.DataFrame(sorted(categories.items()), columns=["category", "count"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(cat_df["category"], cat_df["count"], color="#74c476")
    ax.set_title("Local UNOSAT label inventory by shapefile category")
    ax.set_xlabel("Shapefile category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIG_DIR / "local_label_inventory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    yearly = manifest_df.groupby("year", as_index=False)["scene_count"].sum()
    yearly["unavailable_labels"] = np.where(yearly["year"] == 2024, 1, 0)
    fig, ax1 = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax1.bar(yearly["year"].astype(str), yearly["scene_count"], color="#9ecae1", label="SAR scenes")
    ax1.set_ylabel("SAR scenes")
    ax1.set_xlabel("Year")
    ax1.set_title("Dataset availability summary: SAR coverage vs strong local labels")
    ax2 = ax1.twinx()
    ax2.plot(yearly["year"].astype(str), yearly["unavailable_labels"], color="#de2d26", marker="o", linewidth=2, label="Strong local flood labels")
    ax2.set_ylabel("Label availability flag")
    ax2.set_yticks([0, 1], ["Limited", "Yes"])
    fig.savefig(FIG_DIR / "dataset_availability_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_multi_area_diagrams(config, manifest_df):
    region_order = [region["name"] for region in config["regions"]]
    years = sorted(manifest_df["year"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharey=True)
    axes = axes.flatten()
    palette = ["#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]
    for ax, region_name, color in zip(axes, region_order, palette):
        group = manifest_df[manifest_df["region_name"] == region_name].sort_values("year")
        ax.bar(group["year"].astype(str), group["scene_count"], color=color)
        ax.set_title(region_name)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year")
        ax.set_ylabel("Scenes")
    fig.suptitle("Flood-season Sentinel-1 RTC coverage by region (small multiples)", fontsize=14)
    fig.savefig(FIG_DIR / "scene_count_small_multiples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_2025 = manifest_df[manifest_df["year"] == 2025].sort_values("scene_count", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.pie(df_2025["scene_count"], labels=df_2025["region_name"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Share of 2025 flood-season Sentinel-1 RTC scenes by region")
    fig.savefig(FIG_DIR / "scene_share_2025_pie.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for region_name, group in manifest_df.groupby("region_name"):
        group = group.sort_values("year").copy()
        group["cumulative_scene_count"] = group["scene_count"].cumsum()
        ax.plot(group["year"], group["cumulative_scene_count"], marker="o", linewidth=2, label=region_name)
    ax.set_title("Cumulative flood-season scene growth across study regions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative scenes")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(FIG_DIR / "cumulative_scene_growth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_stations = []
    for region in config["regions"]:
        all_stations.extend(region.get("stations", []))
    station_names = sorted(dict.fromkeys(all_stations))
    station_matrix = np.zeros((len(region_order), len(station_names)), dtype=int)
    region_lookup = {region["name"]: region for region in config["regions"]}
    for i, region_name in enumerate(region_order):
        stations = set(region_lookup[region_name].get("stations", []))
        for j, station in enumerate(station_names):
            station_matrix[i, j] = int(station in stations)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.imshow(station_matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(station_names)), station_names, rotation=30, ha="right")
    ax.set_yticks(range(len(region_order)), region_order)
    ax.set_title("Region-to-gauge station support matrix")
    for i in range(len(region_order)):
        for j in range(len(station_names)):
            ax.text(j, i, "Yes" if station_matrix[i, j] else "-", ha="center", va="center", fontsize=8)
    fig.savefig(FIG_DIR / "station_region_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = manifest_df.groupby("region_name", as_index=False).agg(
        total_scenes=("scene_count", "sum"),
        mean_scenes_per_year=("scene_count", "mean"),
        best_year_scenes=("scene_count", "max"),
    )
    recent_2025 = manifest_df[manifest_df["year"] == 2025][["region_name", "scene_count"]].rename(columns={"scene_count": "scenes_2025"})
    summary = summary.merge(recent_2025, on="region_name", how="left")
    summary["mean_scenes_per_year"] = summary["mean_scenes_per_year"].round(1)
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axis("off")
    table = ax.table(cellText=summary.values, colLabels=summary.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#fee6ce")
            cell.set_text_props(weight="bold")
    ax.set_title("Multi-region scene summary table (2017-2025)", fontsize=13, pad=12)
    fig.savefig(FIG_DIR / "region_scene_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    axes = axes.flatten()
    for ax, region in zip(axes, config["regions"]):
        preview_path = QUICKLOOK_2025_DIR / f"{region['id']}_2025.png"
        if preview_path.exists():
            img = plt.imread(preview_path)
            ax.imshow(img)
            ax.set_title(f"{region['name']} 2025 sample")
        else:
            ax.text(0.5, 0.5, f"No quicklook\nfor {region['name']}", ha="center", va="center", fontsize=12)
            ax.set_title(f"{region['name']} 2025 sample")
        ax.axis("off")
    fig.suptitle("Public 2025 Sentinel-1 quicklook samples across all study areas", fontsize=14)
    fig.savefig(FIG_DIR / "multi_region_quicklooks_2025.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_manifest_file():
    rows = []
    categories = {
        "dataset": {
            "dataset_description_overview.png": "Dataset table overview",
            "study_area_bboxes.png": "Study area bounding boxes",
            "data_source_matrix.png": "Source-to-task matrix",
            "local_label_inventory.png": "Local UNOSAT label inventory",
            "dataset_availability_summary.png": "SAR vs label availability summary",
        },
        "coverage": {
            "scene_count_heatmap.png": "Scene count heatmap",
            "scene_count_by_year.png": "Scene totals by year",
            "scene_count_by_region.png": "Scene totals by region",
            "scene_count_trends.png": "Regional scene trends",
            "study_catalog_timeline.png": "Binary availability timeline",
            "scene_count_2025_only.png": "2025-only coverage chart",
            "scene_count_small_multiples.png": "Region-wise yearly small multiples",
            "scene_share_2025_pie.png": "2025 scene share pie chart",
            "cumulative_scene_growth.png": "Cumulative scene growth curve",
            "region_scene_summary_table.png": "Regional scene summary table",
        },
        "real_data": {
            "real_rtc_vv_vh_sylhet_june2024.png": "Real VV/VH RTC visualization",
            "real_unosat_overlay_sylhet_june2024.png": "UNOSAT overlay on RTC background",
            "sample_sentinel1_quicklook.png": "2024 sample quicklook",
            "sample_sentinel1_quicklook_2025.png": "2025 sample quicklook",
            "multi_region_quicklooks_2025.png": "Six-region 2025 quicklook grid",
        },
        "support": {
            "station_region_matrix.png": "Region-to-gauge station matrix",
        },
        "evaluation": {
            "benchmark_models_matrix.png": "Benchmark model comparison matrix",
            "ablation_training_curves.png": "Ablation training curves",
            "ablation_metrics_bar.png": "Ablation metrics chart",
            "confusion_matrix_casa_net.png": "Confusion matrix",
            "precision_recall_roc_casa_net.png": "PR and ROC diagnostics",
            "uncertainty_sample_patches.png": "Patch uncertainty examples",
        },
    }
    for category, items in categories.items():
        for fig_name, purpose in items.items():
            rows.append(
                {
                    "figure_name": fig_name,
                    "category": category,
                    "purpose": purpose,
                    "status": "generated" if (FIG_DIR / fig_name).exists() else "missing",
                }
            )
    pd.DataFrame(rows).to_csv(FIG_DIR / "evaluation_diagrams_manifest.csv", index=False)


def main():
    ensure_dirs()
    config = ensure_2025_in_config(load_config())
    make_real_rtc_figure()
    make_real_overlay_figure()
    manifest_df = query_scene_coverage(config)
    make_scene_coverage_figures(manifest_df)
    make_dataset_description_figures(config, manifest_df)
    make_multi_area_diagrams(config, manifest_df)
    update_manifest_file()
    print("Updated figures and manifests through 2025.")


if __name__ == "__main__":
    main()
