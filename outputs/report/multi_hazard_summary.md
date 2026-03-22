# Multi-Hazard Summary

- All three hazards now have trained checkpoints, calibrated thresholds, and saved evaluation artifacts.
- Flood remains the strongest externally validated Bangladesh branch and uses the strict recall-tuned benchmark for conservative reporting.
- Landslide now has a six-model CUDA benchmark on the shared multi-hazard split, with CASA-Net ranked first, and it is additionally anchored to an external Bangladesh event catalog from NASA.
- Erosion now has a six-model CUDA benchmark on the shared multi-hazard split, with CASA-Net ranked first.
- Combined maturity table: `outputs/report/multi_hazard_maturity_matrix.csv`
- Combined branch metrics: `outputs/report/multi_hazard_branch_metrics.csv`
- Erosion upper-bound CASA-Net run: Precision 0.4912, Recall 0.4403, F1 0.4644, IoU 0.3024 on the overlap-128 fused active-bank-zone evaluation.