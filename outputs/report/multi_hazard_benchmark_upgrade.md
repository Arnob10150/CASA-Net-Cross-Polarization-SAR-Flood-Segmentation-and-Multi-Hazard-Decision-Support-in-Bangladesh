# Multi-Hazard Benchmark Upgrade

This update promotes the secondary hazards from single-model branch experiments to benchmark-complete branches.

## What is now on disk
- Flood: strict recall-tuned benchmark at `outputs/report/benchmark_results_best_recall_tuned.csv`
- Landslide: six-model CUDA benchmark at `outputs/report/benchmark_results_landslide_secondary.csv`
- Erosion: six-model CUDA benchmark at `outputs/report/benchmark_results_erosion_secondary.csv`
- Combined branch summary: `outputs/report/multi_hazard_branch_metrics.csv`
- Combined maturity matrix: `outputs/report/multi_hazard_maturity_matrix.csv`

## Best saved branch models
- Flood: CASA-Net, Accuracy 0.8046, Precision 0.5367, Recall 0.8650, F1 0.6624, IoU 0.4952
- Landslide: CASA-Net, Accuracy 0.9576, Precision 0.6766, Recall 0.7758, F1 0.7228, IoU 0.5660, ROC-AUC 0.9807
- Erosion: CASA-Net, Accuracy 0.9493, Precision 0.2858, Recall 0.4082, F1 0.3362, IoU 0.2021, ROC-AUC 0.8236

## New figure set
- `outputs/figures/landslide_benchmark_model_comparison.png`
- `outputs/figures/landslide_benchmark_ranked_leaderboard.png`
- `outputs/figures/landslide_confusion_matrix_benchmark.png`
- `outputs/figures/landslide_precision_recall_roc_benchmark.png`
- `outputs/figures/landslide_loss_accuracy_curve_benchmark.png`
- `outputs/figures/erosion_benchmark_model_comparison.png`
- `outputs/figures/erosion_benchmark_ranked_leaderboard.png`
- `outputs/figures/erosion_confusion_matrix_benchmark.png`
- `outputs/figures/erosion_precision_recall_roc_benchmark.png`
- `outputs/figures/erosion_loss_accuracy_curve_benchmark.png`
- `outputs/figures/multi_hazard_branch_metrics.png`
- `outputs/figures/multi_hazard_maturity_scorecard.png`

## Honest scope note
Flood remains the only externally validated branch. Landslide and erosion are now benchmark-complete local Bangladesh branches with the same training and evaluation artifact stack, but their validation source remains local rather than external event reference mapping.


## Updated erosion upper-bound
- `outputs/report/test_metrics_erosion_optimistic_casanet_ft.csv`
- Accuracy 0.9637, Precision 0.4169, Recall 0.4765, F1 0.4447, IoU 0.2860
