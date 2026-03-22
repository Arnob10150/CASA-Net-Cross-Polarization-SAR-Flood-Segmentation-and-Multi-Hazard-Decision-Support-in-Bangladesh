# Erosion Fused Active Bank Zone Summary

This evaluation keeps the current optimistic CUDA erosion CASA-Net checkpoint and scores it against a stronger Bangladesh-focused active-bank-zone target built from the local HydroRIVERS corridor mask plus an independent external optical bankline-change reference.

- Accuracy: `0.9605`
- Precision: `0.4912`
- Recall: `0.4403`
- F1: `0.4644`
- IoU: `0.3024`
- ROC-AUC: `0.9021`
- Average Precision: `0.4224`
- Best threshold: `0.85`

Metrics CSV: `f:\MAPATHON\sylhet_flood_2024\outputs\report\test_metrics_erosion_fused_active_bank_zone_v1_eval_only.csv`
Threshold sweep: `f:\MAPATHON\sylhet_flood_2024\outputs\report\erosion_fused_active_bank_zone_v1_threshold_sweep_eval_only.csv`