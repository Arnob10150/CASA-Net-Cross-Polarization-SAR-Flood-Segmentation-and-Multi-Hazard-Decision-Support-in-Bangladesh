# CASA-Net Bangladesh Flood and Multi-Hazard Upload Package

## What This Folder Contains
This folder is a Git-upload-ready code-and-data package assembled from the working project. It includes the notebooks, analysis scripts, curated datasets, processed label masks, benchmark outputs, and figures needed to review or publish the technical work without copying the entire heavy workspace.

## Folder Structure
- `notebooks/`: the 8 ordered Jupyter notebooks for download, preprocessing, mask preparation, patch generation, CASA-Net training, inference, overlay analysis, and threshold/report generation.
- `analysis/`: Python scripts used for training, benchmark refinement, optimistic/strict evaluation, asset generation, and multi-hazard support.
- `config/`: study-area and hazard catalog JSON files.
- `outputs/report/`: benchmark tables, summary markdown files, branch metrics, label catalogs, scene manifests, and test metrics.
- `outputs/figures/`: curated publication/submission figures and the figure manifest.

## Datasets
- Sylhet 2024 UNOSAT flood extent shapefile components.
- Bangladesh erosion supervision files based on HydroRIVERS corridor labels.
- Bangladesh landslide supervision files and Bangladesh subset of the NASA Global Landslide Catalog.
- Processed flood, landslide, erosion, external optical erosion, and fused active-bank-zone masks.
- Split manifests and normalization metadata for strict and optimistic experiments.
- Scene and label catalogs for the broader 2017-2025 study scope.

Excluded intentionally:
- Large training patch arrays (`.npz` patch stores).
- Large model checkpoints (`.pth` files).
- Large datasets 
- Temporary helper scripts and scratch files.
- Full raw archive dumps beyond the curated label subset.

## Measured Headline Results
- Strict flood benchmark: Accuracy 80.46%, Precision 53.67%, Recall 86.50%, F1 66.24%, IoU 49.52%.
- Optimistic flood benchmark: Accuracy 92.52%, Precision 79.20%, Recall 88.11%, F1 83.42%, IoU 71.55%.
- Landslide branch: Accuracy 95.76%, Precision 67.66%, Recall 77.58%, F1 72.28%, IoU 56.60%, ROC-AUC 0.9807.
- Erosion fused active-bank-zone evaluation: Accuracy 96.05%, Precision 49.12%, Recall 44.03%, F1 46.44%, IoU 30.24%, ROC-AUC 0.9021.

## Notes
- The repo is designed for review and upload, not for re-running the full heavy training pipeline from scratch without re-downloading or regenerating the excluded large assets like dataset.
- Credentials are not included; use `.env.example` as the template.
