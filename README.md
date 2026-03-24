# CASA-Net: Uncertainty-Aware SAR Flood Mapping and Multi-Hazard Decision Support in Bangladesh

> **Mapathon 2026 — Track 01: ResilienceAI · RIMES & Save the Children**
> Team **AIUB_Hydra** · American International University-Bangladesh
> Arnob Aich Anurag (Team Leader) · Shamiul Islam · Tasfia Tasnim

---

## Overview

CASA-Net (**C**ross-polarization **A**ttention **SA**R **A**symmetric **Net**work) is a physically motivated dual-stream deep learning architecture for multi-hazard geospatial mapping in Bangladesh. Rather than concatenating VV and VH polarizations symmetrically, CASA-Net processes them through asymmetric encoders, fuses them via Cross-Polarization Attention Gates (CPAG), and conditions the decoder using terrain-aware FiLM modulation. Monte Carlo Dropout provides epistemic uncertainty estimates alongside every flood map.

The pipeline covers three hazard branches — **flood**, **riverbank erosion**, and **landslide** — and connects segmentation outputs to child-centred anticipatory action through infrastructure exposure analysis and FFWC gauge threshold calibration.

---

## Headline Results

| Branch | Regime | Accuracy | Precision | Recall | F1 | IoU | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Flood** | Strict (non-overlap) | 80.46% | 53.67% | **86.50%** | 66.24% | 49.52% | — |
| **Flood** | Optimistic (overlap) | **92.52%** | **79.20%** | **88.11%** | **83.42%** | **71.55%** | — |
| **Landslide** | Local validation | 95.76% | 67.66% | 77.58% | 72.28% | 56.60% | **0.9807** |
| **Erosion** | Fused bank-zone eval | 96.05% | 49.12% | 44.03% | 46.44% | 30.24% | **0.9021** |

CASA-Net ranks **1st out of 6 models** across all three hazard benchmarks against U-Net, U-Net++, FPN, DeepLabV3+, and MAnet.

> **Benchmark note:** The strict non-overlap regime minimises spatial leakage and is the conservative research reference. The optimistic overlap regime is the competition-facing upper bound. Both are reported separately throughout this repo and should not be merged into a single claim.

---

## Repository Structure

```
casa_net/
│
├── notebooks/                          # 8 ordered Jupyter notebooks — run top-to-bottom
│   ├── 01_data_download.ipynb          # Sentinel-1 API download, UNOSAT, FFWC, HDX, WorldPop
│   ├── 02_sar_preprocessing.ipynb      # Orbit, calibration, speckle filter, terrain correction
│   ├── 03_mask_preparation.ipynb       # UNOSAT rasterisation, JRC water exclusion, class balance
│   ├── 04_patch_generation.ipynb       # 256×256 tiling, VV/VH/terrain stacking, stratified split
│   ├── 05_casa_net_training.ipynb      # Architecture, ablation study, training loop, checkpoints
│   ├── 06_inference_flood_mapping.ipynb# Full-scene inference, flood/uncertainty GeoTIFF export
│   ├── 07_infrastructure_overlay.ipynb # Facility exposure, WorldPop, union-level impact tables
│   └── 08_threshold_and_report.ipynb   # FFWC threshold calibration, anticipatory action matrix
│
├── analysis/                           # Standalone Python scripts
│   ├── train_benchmark.py              # Six-model CUDA benchmark runner
│   ├── benchmark_refinement.py         # Recall-priority threshold calibration
│   ├── optimistic_eval.py              # Overlap-128 optimistic benchmark
│   ├── multi_hazard_support.py         # Erosion and landslide branch training
│   └── asset_generation.py            # Figure and report asset generation
│
├── config/                             # Study-area and hazard catalog definitions
│   ├── study_areas_years.json          # 6 regions × 9 years, bounding boxes, gauge stations
│   └── hazard_catalog.json            # Per-hazard task, inputs, label mode, source URLs
│
├── data/                               # (EXCLUDED — see notes below)
│   ├── raw/
│   │   ├── unosat/                     # Sylhet June 2024 flood extent shapefile components
│   │   ├── erosion/                    # HydroRIVERS corridor supervision labels
│   │   ├── landslide/                  # CHT landslide supervision + NASA GLC Bangladesh subset
│   │   └── masks/                      # Processed flood, landslide, erosion, fused bank-zone masks
│   └── processed/
│       ├── patches/
│       │   ├── split_index.json        # Train/val/test patch assignments (SEED=42, stratified)
│       │   └── norm_stats.json         # Per-channel VV/VH normalisation statistics
│       └── [patch .npz files]          # EXCLUDED — large binary patch arrays not uploaded
│
├── outputs/
│   ├── report/                         # Benchmark tables, catalogs, manifests (CSV + Markdown)
│   │   ├── benchmark_results_best_recall_tuned.csv   # Strict 6-model comparison
│   │   ├── benchmark_results_final.csv               # Optimistic 6-model comparison
│   │   ├── benchmark_results_landslide_secondary.csv # Landslide 6-model comparison
│   │   ├── benchmark_results_erosion_secondary.csv   # Erosion 6-model comparison
│   │   ├── multi_hazard_branch_metrics.csv           # Cross-branch summary table
│   │   ├── multi_hazard_maturity_matrix.csv          # Branch completeness status
│   │   ├── bangladesh_multiyear_flood_label_catalog.csv  # 61 flood shapefiles, 2016–2024
│   │   ├── bangladesh_preferred_hazard_datasets.csv  # Per-hazard preferred dataset registry
│   │   ├── patch_manifest_training_run.csv           # 320-patch index with strat_key
│   │   ├── scene_coverage_manifest.csv               # SAR scene inventory 2017–2025
│   │   ├── training_readiness_status.csv             # Branch readiness checklist
│   │   └── test_metrics_erosion_fused_active_bank_zone_v1_eval_only.csv
│   │
│   └── figures/                        # Publication-ready figures (PNG)
│       ├── study_area_bboxes_fixed.png
│       ├── benchmark_model_comparison_final.png
│       ├── real_unosat_overlay_sylhet_june2024.png
│       ├── regional_sentinel1_quicklook_triptych.png
│       ├── multi_hazard_branch_metrics.png
│       ├── multi_hazard_maturity_scorecard.png
│       ├── multi_hazard_pipeline_framework.png
│       ├── bangladesh_preferred_hazard_datasets.png
│       ├── landslide_benchmark_model_comparison.png
│       ├── landslide_benchmark_ranked_leaderboard.png
│       ├── landslide_confusion_matrix_benchmark.png
│       ├── landslide_precision_recall_roc_benchmark.png
│       ├── landslide_loss_accuracy_curve_benchmark.png
│       ├── bangladesh_external_landslide_catalog_map.png
│       ├── bangladesh_external_landslide_catalog_by_year.png
│       ├── erosion_benchmark_model_comparison.png
│       ├── erosion_benchmark_ranked_leaderboard.png
│       ├── erosion_confusion_matrix_benchmark.png
│       ├── erosion_precision_recall_roc_benchmark.png
│       ├── erosion_loss_accuracy_curve_benchmark.png
│       ├── erosion_fused_active_bank_zone_preview.png
│       ├── erosion_branch_improvement_summary.png
│       └── erosion_fused_active_bank_zone_threshold_tradeoff.png
│
├── models/                             # EXCLUDED — .pth checkpoints not uploaded (large)
│
├── .env.example                        # Credential template (Copernicus, FFWC API keys)
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # This file
```

---

## Datasets

### Primary SAR Imagery

| Dataset | Format | Coverage | Years | Role | Source |
|---|---|---|---|---|---|
| **Sentinel-1 RTC** | GRD, VV + VH | 6 study regions | 2017–2025 | SAR input backbone | [Copernicus Data Space](https://dataspace.copernicus.eu/) |

**Study regions and gauge stations:**

| Region | Bounding Box (lon, lat) | FFWC Gauge Station(s) | Danger Level |
|---|---|---|---|
| Sylhet | 91.6–92.5°E, 24.5–25.2°N | Kanaighat, Sylhet, Sheola | 12.75 m / 10.80 m / 13.05 m |
| Sunamganj | 90.9–91.7°E, 24.7–25.3°N | Sunamganj | 7.80 m |
| Kurigram | 89.4–89.9°E, 25.4–26.1°N | Chilmari | — |
| Gaibandha | 89.1–89.7°E, 25.0–25.6°N | Fulchhari | — |
| Jamalpur | 89.7–90.2°E, 24.6–25.3°N | Bahadurabad | — |
| Sirajganj | 89.4–89.9°E, 24.1–24.9°N | Sirajganj | — |

**Temporal windows tracked per year:**

```
pre_flood:   May 10 – May 25
peak_flood:  June 15 – August 31
post_flood:  September 10 – September 25
```

---

### Flood Branch Datasets

| Dataset | Format | Coverage | Role | Source |
|---|---|---|---|---|
| **UNOSAT Sylhet June 2024 flood extent** | Shapefile (polygon) | Sylhet division | Primary flood ground truth (training + evaluation) | [HDX / UNOSAT FL20240502BGD](https://data.humdata.org/) |
| **Bangladesh multi-year flood label catalog** | CSV index + shapefiles | National, 2016–2024 | 61 flood/water extent shapefiles across 5 archived event folders | HDX, Copernicus EMS, DFO, UNDP GeoHub |
| **FFWC river gauge data** | Time series CSV | Kanaighat, Sylhet, Sheola, Sunamganj | Anticipatory action threshold calibration | [FFWC / BWDB API](https://ffwc.bwdb.gov.bd/data_load/) |
| **SRTM 30m DEM** | GeoTIFF raster | AOI-aligned | Terrain correction + slope, HAND derivation | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) / [OpenTopography](https://opentopography.org/) |
| **JRC Global Surface Water** | GeoTIFF raster | AOI-aligned | Permanent water prior (HTC branch input) | [JRC GSW](https://global-surface-water.appspot.com/) |
| **WorldPop Bangladesh 100m** | GeoTIFF raster | Bangladesh | Population exposure estimation | [WorldPop Hub](https://hub.worldpop.org/geodata/summary?id=94) |
| **HDX Infrastructure layers** | GeoJSON (points/polygons) | Bangladesh | Schools, hospitals, cyclone shelters, admin boundaries | [HDX Bangladesh](https://data.humdata.org/group/bgd) |

**Flood model inputs per patch:**
```
vv      → shape (1, 256, 256)   VV backscatter in dB
vh      → shape (1, 256, 256)   VH backscatter in dB
terrain → shape (4, 256, 256)   [slope, TWI, JRC water prior, HAND]
mask    → shape (1, 256, 256)   binary flood ground truth
```

**Normalisation statistics (saved from training set):**
```
vv_mean: -11.0708   vv_std: 6.6899
vh_mean: -16.8866   vh_std: 5.5918
```

**Patch inventory:**
```
Total patches:           320
With flood pixels:       160  (50%)
Without flood pixels:    160  (50%)
Mean flood-pixel fraction: 0.1430
Stratification bins:     zero=160, high=118, medium=23, low=19
Train / Val / Test:      224 / 48 / 48  (70% / 15% / 15%)
Split file:              data/processed/patches/split_index.json
```

---

### Erosion Branch Datasets

| Dataset | Format | Coverage | Role | Source |
|---|---|---|---|---|
| **HydroRIVERS corridor supervision** | Raster (corridor seed) | Bangladesh (Jamuna focus) | Primary erosion label seed | [HydroSHEDS](https://www.hydrosheds.org/products) |
| **JRC GSW Yearly Water History** | GeoTIFF raster | AOI-aligned | Annual water extent change for bankline detection | [JRC GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory) |
| **External optical bankline reference** | Raster | Bangladesh rivers | Fused active-bank-zone target construction | NHESS 2023 + [Zenodo 7253121](https://doi.org/10.5281/zenodo.7253121) / [7252970](https://doi.org/10.5281/zenodo.7252970) |
| **Fused active bank zone mask** | GeoTIFF raster | Bangladesh AOIs | Stronger evaluation target (HydroRIVERS + optical) | Derived locally |

**Preferred reference:** *Assessing riverbank erosion in Bangladesh using time series of Sentinel-1 radar imagery in the Google Earth Engine* (NHESS 2023) — [doi:10.5194/nhess-23-751-2023](https://nhess.copernicus.org/articles/23/751/2023/)

**Erosion model inputs:** `sentinel1_vv`, `sentinel1_vh`, `hydrorivers`, `jrc_yearly_history`, `distance_to_river`, `bankfull_zone`

---

### Landslide Branch Datasets

| Dataset | Format | Coverage | Role | Source |
|---|---|---|---|---|
| **Landslide Inventory (2001–2017) — Chittagong Hilly Areas** | Point/polygon | Chittagong Hill Tracts | Primary landslide label reference | [MDPI Data 2019](https://www.mdpi.com/2306-5729/5/1/4) |
| **NASA Global Landslide Catalog — Bangladesh subset** | CSV / point | Bangladesh | External event-point validation reference | [COOLR / NASA](https://gpm.nasa.gov/applications/landslides/coolr) · [data.gov](https://catalog.data.gov/dataset/global-landslide-catalog-export) |
| **CHIRPS rainfall** | Raster time series | Regional | Rainfall trigger forcing | [UCSB CHC](https://www.chc.ucsb.edu/data/chirps) |
| **GPM IMERG** | Raster time series | Regional | Precipitation forcing | [NASA GPM](https://gpm.nasa.gov/) |
| **Terrain derivatives** | GeoTIFF rasters | AOI-aligned | Slope, curvature, roughness (from SRTM DEM) | Derived locally via richdem / pysheds |

**Landslide model inputs:** `slope`, `curvature`, `roughness`, `chirps_rainfall`, `gpm_imerg`, `coolr_landslides`, `landcover`

---

## Architecture: CASA-Net

```
Sentinel-1 VV ──► ResNet34 Encoder ─────────────────────────────────────┐
                  [F1, F2, F3, F4]                                       │
                        │                                                 │
              Cross-Polarization Attention Gate (CPAG)           U-Net Decoder
              Q(VV) · K(VH)ᵀ / √d  ◄──────────────────────────  + FiLM + MC Dropout
                        │                                                 │
Sentinel-1 VH ──► MobileNetV3 Encoder ───────────────────────────────────┘
                  [G1, G2, G3, G4]

DEM + Slope ─────────────────────────► Haor Terrain Conditioning Branch
TWI + HAND + JRC Water               (3-layer CNN → FiLM: γ(terrain)·x + β)
                                              │
                                       Applied to all decoder stages

Outputs:
  ├── Flood probability map    (0–1, continuous)
  ├── Epistemic uncertainty map (std dev across 10 MC Dropout passes)
  └── Binary flood extent       (threshold=0.50 → GeoTIFF)
```

**Novel components:**

| Component | Description |
|---|---|
| **Asymmetric encoders** | ResNet34 for VV (open water), MobileNetV3 for VH (flooded vegetation) |
| **CPAG** | Cross-Polarization Attention Gate — cross-modal attention between VV and VH streams |
| **FiLM HTC** | Haor Terrain Conditioning — slope, TWI, HAND, JRC prior modulate decoder features |
| **MC Dropout** | 10 inference passes → mean probability + epistemic uncertainty map |

**Training setup:**

```
Loss:       BCEDiceLoss = 0.5 × BCE + 0.5 × (1 − Dice)
Optimizer:  AdamW  (lr=1e-4, weight_decay=1e-5)
Scheduler:  CosineAnnealingLR  (T_max=50)
Seed:       SEED = 42  (torch, numpy, random — all fixed)
Patch size: 256 × 256 px,  64 px overlap
```

**Ablation variants trained:**

| Variant | CPAG | HTC (FiLM) | MC Dropout |
|---|---|---|---|
| Baseline U-Net | ✗ | ✗ | ✗ |
| + CPAG only | ✓ | ✗ | ✗ |
| + HTC only | ✗ | ✓ | ✗ |
| **CASA-Net (full)** | ✓ | ✓ | ✓ |

---

## Setup and Usage

### 1. Clone and install

```bash
git clone https://github.com/your-repo/casa-net-bangladesh.git
cd casa-net-bangladesh
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and add:
#   COPERNICUS_USER=your_email
#   COPERNICUS_PASSWORD=your_password
```

### 3. Run notebooks in order

```bash
jupyter notebook notebooks/01_data_download.ipynb
# Then 02, 03, 04, 05, 06, 07, 08 in sequence
```

Each notebook is self-contained and saves all intermediate outputs to disk, so any notebook can be re-run independently after its prerequisites exist.

---

## Key Output Files

| File | Description |
|---|---|
| `outputs/report/benchmark_results_best_recall_tuned.csv` | Strict 6-model flood benchmark |
| `outputs/report/benchmark_results_final.csv` | Optimistic 6-model flood benchmark |
| `outputs/report/benchmark_results_landslide_secondary.csv` | Landslide 6-model benchmark |
| `outputs/report/benchmark_results_erosion_secondary.csv` | Erosion 6-model benchmark |
| `outputs/report/multi_hazard_branch_metrics.csv` | Cross-hazard summary |
| `outputs/report/bangladesh_multiyear_flood_label_catalog.csv` | 61 flood label shapefiles, 2016–2024 |
| `outputs/report/patch_manifest_training_run.csv` | 320-patch training index |
| `outputs/report/scene_coverage_manifest.csv` | SAR scene count by region × year |
| `config/study_areas_years.json` | Study region definitions |
| `config/hazard_catalog.json` | Multi-hazard input/output/supervision catalog |
| `data/processed/patches/split_index.json` | Reproducible train/val/test assignments |

---

## What Is Not Included

The following are intentionally excluded from this upload to keep the repository lightweight:

| Excluded | Reason |
|---|---|
| `data` | Large binary patch arrays (~GB range) along with large files and images |
| `models/*.pth` | Trained model checkpoints (~100–500 MB each) |
| Full raw Sentinel-1 `.SAFE` scenes | Very large (multi-GB per scene) |
| Full raw DEM tiles | Available from USGS EarthExplorer on demand |
| Temporary and scratch helper scripts | Not needed for review or reproduction |

To reproduce from scratch, follow the notebooks in order starting from `01_data_download.ipynb`. The `.env.example` file documents all required credentials.

---

## Citation

If you use this work, please cite:

```
Anurag, A. A., Islam, S., & Tasnim, T. (2026). CASA-Net: Uncertainty-Aware SAR Flood
Mapping and Multi-Hazard Decision Support in Bangladesh. Mapathon 2026, RIMES &
Save the Children. Team AIUB_Hydra, American International University-Bangladesh.
```

---

## License

MIT License. See `LICENSE` for details.

---

*Organized by RIMES & Save the Children · Funded by GFFO · Mapathon 2026*
