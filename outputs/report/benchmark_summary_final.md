# Optimistic CUDA Benchmark Summary

This benchmark uses an overlap-prone random patch split and is intentionally optimistic.

| Model      |   Accuracy |   Precision |   Recall |       F1 |      IoU |     Dice |   Threshold |   BestValIoU |   BestEpoch |   Params | Checkpoint                                                                  |   Rank |
|:-----------|-----------:|------------:|---------:|---------:|---------:|---------:|------------:|-------------:|------------:|---------:|:----------------------------------------------------------------------------|-------:|
| CASA-Net   |   0.925163 |    0.791965 | 0.881098 | 0.834157 | 0.715497 | 0.834157 |        0.75 |     0.727212 |          13 | 25688081 | f:\MAPATHON\sylhet_flood_2024\models\casa_net_best_optimistic_tuned.pth     |      1 |
| U-Net++    |   0.867497 |    0.652778 | 0.811144 | 0.723395 | 0.566655 | 0.723395 |        0.65 |     0.57732  |           6 | 26075473 | f:\MAPATHON\sylhet_flood_2024\models\benchmark_unetplusplus_optimistic.pth  |      2 |
| U-Net      |   0.863268 |    0.640953 | 0.818249 | 0.71883  | 0.561073 | 0.71883  |        0.65 |     0.572075 |           6 | 24433233 | f:\MAPATHON\sylhet_flood_2024\models\benchmark_unet_optimistic.pth          |      3 |
| DeepLabV3+ |   0.862433 |    0.64027  | 0.812439 | 0.716152 | 0.557817 | 0.716152 |        0.65 |     0.577676 |           6 | 22434321 | f:\MAPATHON\sylhet_flood_2024\models\benchmark_deeplabv3plus_optimistic.pth |      4 |
| FPN        |   0.850425 |    0.612228 | 0.817625 | 0.700174 | 0.538668 | 0.700174 |        0.55 |     0.552944 |           5 | 23152257 | f:\MAPATHON\sylhet_flood_2024\models\benchmark_fpn_optimistic.pth           |      5 |
| MAnet      |   0.838329 |    0.5897   | 0.799198 | 0.678649 | 0.513602 | 0.678649 |        0.7  |     0.532093 |           6 | 31780497 | f:\MAPATHON\sylhet_flood_2024\models\benchmark_manet_optimistic.pth         |      6 |

Top model: CASA-Net
Top IoU: 0.7155
Top F1: 0.8342