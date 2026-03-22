from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(r"f:\MAPATHON\sylhet_flood_2024")
HAZARD_CATALOG_PATH = ROOT / "config" / "hazard_catalog.json"


def load_hazard_catalog(path: Path | None = None) -> dict:
    path = path or HAZARD_CATALOG_PATH
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ordered_hazard_ids(catalog: Mapping) -> List[str]:
    hazards = sorted(catalog.get("hazards", []), key=lambda h: h.get("priority", 999))
    return [hazard["id"] for hazard in hazards]


def ensure_multihazard_tree(root: Path, catalog: Mapping) -> Dict[str, Path]:
    paths = {
        "raw_hazards": root / "data" / "raw" / "hazards",
        "processed_masks": root / "data" / "processed" / "masks_multihazard",
        "processed_features": root / "data" / "processed" / "hazard_features",
        "processed_patches": root / "data" / "processed" / "patches_multihazard_ready",
        "report": root / "outputs" / "report",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for hazard in ordered_hazard_ids(catalog):
        (paths["raw_hazards"] / hazard / "labels").mkdir(parents=True, exist_ok=True)
        (paths["raw_hazards"] / hazard / "auxiliary").mkdir(parents=True, exist_ok=True)
        (paths["processed_masks"] / hazard).mkdir(parents=True, exist_ok=True)
        (paths["processed_features"] / hazard).mkdir(parents=True, exist_ok=True)
        (root / "outputs" / "maps" / hazard).mkdir(parents=True, exist_ok=True)
    return paths


def build_hazard_target_stack(npz: Mapping[str, np.ndarray], hazard_order: Sequence[str]) -> np.ndarray:
    if "hazard_targets" in npz:
        return np.asarray(npz["hazard_targets"], dtype=np.float32)
    if "mask" in npz:
        mask = np.asarray(npz["mask"], dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]
        target = np.zeros((len(hazard_order), mask.shape[-2], mask.shape[-1]), dtype=np.float32)
        if "flood" in hazard_order:
            target[hazard_order.index("flood")] = mask
        return target
    raise KeyError("Expected either 'hazard_targets' or 'mask' in patch payload")


def convert_patch_dataset_to_multihazard(
    source_patch_dir: Path,
    target_patch_dir: Path,
    hazard_order: Sequence[str],
) -> pd.DataFrame:
    source_patch_dir = Path(source_patch_dir)
    target_patch_dir = Path(target_patch_dir)
    target_patch_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patch_path in sorted(source_patch_dir.glob("patch_*.npz")):
        with np.load(patch_path) as npz:
            payload = {key: npz[key] for key in npz.files if key != "hazard_targets"}
            hazard_targets = build_hazard_target_stack(npz, hazard_order)
        target_path = target_patch_dir / patch_path.name
        payload["hazard_targets"] = hazard_targets.astype(np.float32)
        payload["hazard_names"] = np.array(hazard_order, dtype="U32")
        np.savez_compressed(target_path, **payload)
        row = {
            "path": str(target_path),
            "hazard_order": ",".join(hazard_order),
            "flood_positive_px": float(hazard_targets[hazard_order.index("flood")].sum()) if "flood" in hazard_order else 0.0,
        }
        for idx, hazard in enumerate(hazard_order):
            row[f"{hazard}_positive_px"] = float(hazard_targets[idx].sum())
        rows.append(row)

    split_src = source_patch_dir / "split_index.json"
    if split_src.exists():
        split_index = json.loads(split_src.read_text(encoding="utf-8"))
        updated = {}
        for split, paths in split_index.items():
            updated[split] = [str(target_patch_dir / Path(path).name) for path in paths]
        (target_patch_dir / "split_index.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")

    for name in ["norm_stats.json", "patch_manifest.csv", "dataset_info.json"]:
        src = source_patch_dir / name
        if src.exists():
            dst = target_patch_dir / name
            if name.endswith('.json'):
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                dst.write_bytes(src.read_bytes())

    manifest = pd.DataFrame(rows)
    manifest.to_csv(target_patch_dir / "multi_hazard_patch_manifest.csv", index=False)
    return manifest


def build_combined_risk_layer(
    hazard_probabilities: Mapping[str, np.ndarray],
    hazard_weights: Mapping[str, float] | None = None,
    exposure: np.ndarray | None = None,
    vulnerability: np.ndarray | None = None,
) -> np.ndarray:
    catalog = load_hazard_catalog()
    default_weights = catalog.get("combined_risk", {}).get("default_hazard_weights", {})
    hazard_weights = dict(default_weights if hazard_weights is None else hazard_weights)
    first = next(iter(hazard_probabilities.values()))
    combined = np.zeros_like(first, dtype=np.float32)
    total_w = 0.0
    for hazard, arr in hazard_probabilities.items():
        weight = float(hazard_weights.get(hazard, 1.0))
        combined += weight * np.asarray(arr, dtype=np.float32)
        total_w += weight
    combined = combined / max(total_w, 1e-6)
    if exposure is not None:
        combined = combined * np.asarray(exposure, dtype=np.float32)
    if vulnerability is not None:
        combined = combined * np.asarray(vulnerability, dtype=np.float32)
    return np.clip(combined, 0, 1)


class MultiHazardProbabilityHeads(nn.Module):
    def __init__(self, in_channels: int, hazard_names: Sequence[str], dropout_p: float = 0.2):
        super().__init__()
        self.hazard_names = list(hazard_names)
        self.dropout = nn.Dropout2d(dropout_p)
        self.heads = nn.ModuleDict({hazard: nn.Conv2d(in_channels, 1, 1) for hazard in self.hazard_names})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(x)
        return {hazard: torch.sigmoid(head(x)) for hazard, head in self.heads.items()}


def multi_hazard_bcedice_loss(predictions: Mapping[str, torch.Tensor], targets: torch.Tensor, hazard_order: Sequence[str], hazard_weights: Mapping[str, float] | None = None) -> torch.Tensor:
    hazard_weights = dict(hazard_weights or {})
    total = None
    weight_sum = 0.0
    for idx, hazard in enumerate(hazard_order):
        pred = torch.clamp(predictions[hazard], 1e-6, 1 - 1e-6)
        target = targets[:, idx:idx+1]
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
        smooth = 1e-6
        inter = (pred * target).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        loss = 0.5 * bce + 0.5 * dice.mean()
        weight = float(hazard_weights.get(hazard, 1.0))
        total = loss * weight if total is None else total + loss * weight
        weight_sum += weight
    return total / max(weight_sum, 1e-6)
