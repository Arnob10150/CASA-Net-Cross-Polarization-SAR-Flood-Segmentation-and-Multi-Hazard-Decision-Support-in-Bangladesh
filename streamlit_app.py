"""
CASA-Net Streamlit Application
Cross-polarization Attention SAR Asymmetric Network for Flood Segmentation
Supports VV and VH polarization SAR data, multi-hazard predictions,
and Monte Carlo Dropout uncertainty estimation.
"""

import io
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Environment / device configuration
# ---------------------------------------------------------------------------

FORCE_CPU = os.environ.get("CASA_FORCE_CPU", "0") == "1"
DEVICE = torch.device("cpu") if FORCE_CPU else torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_HAZARD_MODE = os.environ.get("MULTI_HAZARD_MODE", "flood").split(",")
MULTI_HAZARD_MODE = [h.strip() for h in MULTI_HAZARD_MODE if h.strip()]

MC_PASSES = 10          # Monte Carlo Dropout inference passes
DEFAULT_THRESHOLD = 0.5
PATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Model architecture (mirrors analysis/train_casanet_optimistic_tuned.py)
# ---------------------------------------------------------------------------

class CPAG(nn.Module):
    """Cross-Polarization Attention Gate."""

    def __init__(self, vv_channels, vh_channels, inter_channels, attn_size=8):
        super().__init__()
        self.q = nn.Conv2d(vv_channels, inter_channels, 1)
        self.k = nn.Conv2d(vh_channels, inter_channels, 1)
        self.v = nn.Conv2d(vh_channels, vh_channels, 1)
        self.proj = nn.Conv2d(vh_channels, vv_channels, 1)
        self.scale = inter_channels ** -0.5
        self.attn_size = attn_size

    def forward(self, f_vv, g_vh):
        if g_vh.shape[-2:] != f_vv.shape[-2:]:
            g_vh = F.interpolate(g_vh, size=f_vv.shape[-2:], mode="bilinear", align_corners=False)
        vv_small = F.adaptive_avg_pool2d(f_vv, self.attn_size)
        vh_small = F.adaptive_avg_pool2d(g_vh, self.attn_size)
        q = self.q(vv_small).flatten(2)
        k = self.k(vh_small).flatten(2)
        v = self.v(vh_small).flatten(2)
        b, _, hw = q.shape
        side = int(hw ** 0.5)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, -1, side, side)
        out = self.proj(out)
        out = F.interpolate(out, size=f_vv.shape[-2:], mode="bilinear", align_corners=False)
        return f_vv + out


class HaorTerrainBranch(nn.Module):
    """Terrain conditioning via FiLM modulation."""

    def __init__(self, in_channels=4, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.generators = nn.ModuleList([nn.Linear(64, 2 * c) for c in decoder_channels])

    def forward(self, terrain):
        context = self.encoder(terrain).flatten(1)
        return [gen(context) for gen in self.generators]


def apply_film(x, params):
    gamma, beta = params.chunk(2, dim=1)
    gamma = 1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
    beta = 0.1 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
    return gamma * x + beta


class ResNet34Encoder(nn.Module):
    """VV-stream encoder: ResNet34 adapted to single-channel input."""

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet34(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(self.maxpool(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class MobileNetV3Encoder(nn.Module):
    """VH-stream encoder: MobileNetV3-Small adapted to single-channel input."""

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = backbone.features

    def forward(self, x):
        outs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in {1, 3, 6, 11}:
                outs.append(x)
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(512, 256, 256),
            DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64),
        ])
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

    def forward(self, fused, film_params):
        x = self.blocks[0](fused[-1], fused[-2])
        x = apply_film(x, film_params[0])
        x = self.blocks[1](x, fused[-3])
        x = apply_film(x, film_params[1])
        x = self.blocks[2](x, fused[-4])
        x = apply_film(x, film_params[2])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final(x)
        x = apply_film(x, film_params[3])
        return x


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels=32, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_p)
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(self.dropout(x)))


class CASANet(nn.Module):
    """Full CASA-Net model."""

    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.vv_encoder = ResNet34Encoder()
        self.vh_encoder = MobileNetV3Encoder()
        self.cpag = nn.ModuleList([
            CPAG(64, 16, 32),
            CPAG(128, 24, 64),
            CPAG(256, 40, 128),
            CPAG(512, 96, 256),
        ])
        self.terrain_branch = HaorTerrainBranch()
        self.decoder = UNetDecoder()
        self.head = UncertaintyHead(32, dropout_p)

    def forward(self, x_vv, x_vh, x_terrain):
        size = x_vv.shape[-2:]
        vv_feats = self.vv_encoder(x_vv)
        vh_feats = self.vh_encoder(x_vh)
        fused = [self.cpag[i](vv_feats[i], vh_feats[i]) for i in range(4)]
        film = self.terrain_branch(x_terrain)
        x = self.decoder(fused, film)
        x = self.head(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def robust_norm(arr: np.ndarray) -> np.ndarray:
    """Clip to [1st, 99th] percentile and scale to [0, 1]."""
    arr = arr.astype(np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_array_from_upload(file_obj, label: str) -> np.ndarray:
    """
    Load a 2-D or 3-D numpy array from an uploaded file.

    Supported formats
    -----------------
    * ``.npy`` – numpy binary
    * ``.npz`` – numpy archive (uses key matching *label* or first key)
    * ``.tif`` / ``.tiff`` – GeoTIFF via rasterio (band 1)
    * ``.nc`` – NetCDF via scipy.io.netcdf (first variable)
    """
    suffix = Path(file_obj.name).suffix.lower()
    data = file_obj.read()

    if suffix == ".npy":
        arr = np.load(io.BytesIO(data))

    elif suffix == ".npz":
        npz = np.load(io.BytesIO(data))
        keys = list(npz.files)
        key = label if label in keys else keys[0]
        arr = npz[key]

    elif suffix in {".tif", ".tiff"}:
        try:
            import rasterio
            from rasterio.io import MemoryFile
            with MemoryFile(data) as memfile:
                with memfile.open() as dataset:
                    arr = dataset.read(1).astype(np.float32)
        except ImportError:
            st.error("rasterio is required to read GeoTIFF files.")
            return None

    elif suffix == ".nc":
        try:
            import netCDF4 as nc_lib
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                ds = nc_lib.Dataset(tmp_path, "r")
                var_names = [k for k in ds.variables if ds.variables[k].ndim >= 2]
                if not var_names:
                    st.error("No 2-D variables found in the NetCDF file.")
                    ds.close()
                    return None
                arr = np.array(ds.variables[var_names[-1]][:], dtype=np.float32)
                ds.close()
            finally:
                os.unlink(tmp_path)
        except ImportError:
            st.error("netCDF4 is required to read NetCDF files. Install it with: pip install netCDF4")
            return None

    else:
        st.error(f"Unsupported file format: {suffix}")
        return None

    arr = arr.squeeze()
    if arr.ndim != 2:
        st.error(f"Expected a 2-D array for {label}, got shape {arr.shape}.")
        return None
    return arr.astype(np.float32)


def make_dummy_terrain(h: int, w: int) -> np.ndarray:
    """Return a zero-filled 4-channel terrain array (slope, TWI, HAND, JRC)."""
    return np.zeros((4, h, w), dtype=np.float32)


def preprocess_sar(vv: np.ndarray, vh: np.ndarray):
    """
    Robust normalise each polarisation band to [0, 1] and
    return tensors shaped (1, 1, H, W) ready for the model.
    """
    vv_n = robust_norm(vv)
    vh_n = robust_norm(vh)
    t_vv = torch.tensor(vv_n[None, None, ...], dtype=torch.float32)
    t_vh = torch.tensor(vh_n[None, None, ...], dtype=torch.float32)
    return t_vv, t_vh


@st.cache_resource(show_spinner="Loading CASA-Net model …")
def load_model(model_path: str | None = None) -> CASANet:
    """Instantiate CASANet and optionally load weights from *model_path*."""
    model = CASANet(dropout_p=0.1)
    if model_path and Path(model_path).exists():
        try:
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            # weights_only parameter not available in PyTorch < 2.0
            state = torch.load(model_path, map_location=DEVICE)  # noqa: S614
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        st.sidebar.success(f"Weights loaded from `{Path(model_path).name}`")
    else:
        st.sidebar.info("No pre-trained weights found – running with random initialisation.")
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(
    model: CASANet,
    t_vv: torch.Tensor,
    t_vh: torch.Tensor,
    t_terrain: torch.Tensor,
    mc_passes: int = MC_PASSES,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Monte Carlo Dropout inference.

    Returns
    -------
    prob_mean : np.ndarray (H, W)  – mean flood probability
    prob_std  : np.ndarray (H, W)  – epistemic uncertainty (std across passes)
    mask_bin  : np.ndarray (H, W)  – binary flood mask at *threshold*
    """
    t_vv = t_vv.to(DEVICE)
    t_vh = t_vh.to(DEVICE)
    t_terrain = t_terrain.to(DEVICE)

    # Enable dropout at inference time for MC sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            out = model(t_vv, t_vh, t_terrain)
            preds.append(out.squeeze().cpu().numpy())

    # Restore eval mode
    model.eval()

    stack = np.stack(preds, axis=0)   # (mc_passes, H, W)
    prob_mean = stack.mean(axis=0)
    prob_std = stack.std(axis=0)
    mask_bin = (prob_mean >= threshold).astype(np.uint8)
    return prob_mean, prob_std, mask_bin


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def colorise_mask(mask: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """
    Blend the binary *mask* overlay on a grayscale background derived from *prob*.

    Returns an (H, W, 3) uint8 RGB image.
    """
    bg = (prob * 255).astype(np.uint8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    # Flood pixels → blue
    rgb[mask == 1] = [30, 100, 220]
    return rgb


def fig_to_pil(fig: plt.Figure):
    """Convert a matplotlib Figure to a PIL Image (via buffer)."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return Image.open(buf)


def plot_results(
    vv: np.ndarray,
    vh: np.ndarray,
    prob_mean: np.ndarray,
    prob_std: np.ndarray,
    mask_bin: np.ndarray,
    hazard_label: str = "Flood",
) -> plt.Figure:
    """Create a 2×3 summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"CASA-Net — {hazard_label} Segmentation", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(robust_norm(vv), cmap="gray")
    axes[0, 0].set_title("VV Polarisation (input)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(robust_norm(vh), cmap="gray")
    axes[0, 1].set_title("VH Polarisation (input)")
    axes[0, 1].axis("off")

    im_prob = axes[0, 2].imshow(prob_mean, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title(f"{hazard_label} Probability Map")
    axes[0, 2].axis("off")
    fig.colorbar(im_prob, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(mask_bin, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Binary {hazard_label} Mask")
    axes[1, 0].axis("off")

    im_std = axes[1, 1].imshow(prob_std, cmap="viridis", vmin=0)
    axes[1, 1].set_title("Uncertainty (MC Dropout std)")
    axes[1, 1].axis("off")
    fig.colorbar(im_std, ax=axes[1, 1], fraction=0.046, pad=0.04)

    overlay = colorise_mask(mask_bin, prob_mean)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay (blue = flooded)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit pages
# ---------------------------------------------------------------------------

def page_inference():
    st.header("🌊 Flood Segmentation Inference")
    st.markdown(
        """
        Upload **VV** and **VH** SAR backscatter images to run flood segmentation
        with the CASA-Net model.  
        Accepted formats: `.npy`, `.npz`, `.tif` / `.tiff`, `.nc`
        """
    )

    # --- sidebar settings ---------------------------------------------------
    st.sidebar.subheader("⚙️ Inference Settings")
    threshold = st.sidebar.slider(
        "Flood probability threshold", min_value=0.1, max_value=0.9,
        value=DEFAULT_THRESHOLD, step=0.05,
    )
    mc_passes = st.sidebar.slider(
        "MC Dropout passes (uncertainty)", min_value=1, max_value=30,
        value=MC_PASSES, step=1,
    )
    hazard_labels = {
        "flood": "Flood",
        "erosion": "Erosion",
        "landslide": "Landslide",
    }
    active_hazards = [hazard_labels.get(h, h.title()) for h in MULTI_HAZARD_MODE]
    hazard_label = st.sidebar.selectbox(
        "Hazard type (display label)",
        options=active_hazards,
        index=0,
    )

    # --- file upload --------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        vv_file = st.file_uploader(
            "📡 VV Polarisation (co-pol)",
            type=["npy", "npz", "tif", "tiff", "nc"],
            key="vv",
        )
    with col2:
        vh_file = st.file_uploader(
            "📡 VH Polarisation (cross-pol)",
            type=["npy", "npz", "tif", "tiff", "nc"],
            key="vh",
        )

    # Optional: model weights file
    model_weights_file = st.file_uploader(
        "🏋️ Model weights (optional, `.pth` / `.pt`)",
        type=["pth", "pt"],
        key="weights",
    )

    # Save uploaded weights to a temp file so load_model can read it
    weights_path = None
    if model_weights_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(model_weights_file.read())
            weights_path = tmp.name

    if st.button("🚀 Run Inference", type="primary", disabled=(vv_file is None or vh_file is None)):
        with st.spinner("Processing …"):
            # Load arrays
            vv_arr = load_array_from_upload(vv_file, "vv")
            vh_arr = load_array_from_upload(vh_file, "vh")
            if vv_arr is None or vh_arr is None:
                st.error("Could not read one or both input files. See error above.")
                return

            if vv_arr.shape != vh_arr.shape:
                st.warning(
                    f"VV shape {vv_arr.shape} ≠ VH shape {vh_arr.shape}. "
                    "Resizing VH to match VV."
                )
                from PIL import Image
                vh_img = Image.fromarray(vh_arr)
                vh_arr = np.array(vh_img.resize((vv_arr.shape[1], vv_arr.shape[0]), Image.BILINEAR))

            # Preprocess
            t_vv, t_vh = preprocess_sar(vv_arr, vh_arr)
            h, w = vv_arr.shape
            terrain_np = make_dummy_terrain(h, w)
            t_terrain = torch.tensor(terrain_np[None, ...], dtype=torch.float32)

            # Load / cache model
            model = load_model(weights_path)

            # Inference
            prob_mean, prob_std, mask_bin = run_inference(
                model, t_vv, t_vh, t_terrain,
                mc_passes=mc_passes,
                threshold=threshold,
            )

        # ---- display metrics -----------------------------------------------
        flood_px = int(mask_bin.sum())
        total_px = mask_bin.size
        flood_pct = 100.0 * flood_px / total_px

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flooded pixels", f"{flood_px:,}")
        m2.metric("Coverage (%)", f"{flood_pct:.2f}%")
        m3.metric("Mean probability", f"{prob_mean.mean():.3f}")
        m4.metric("Mean uncertainty", f"{prob_std.mean():.4f}")

        # ---- visualisation -------------------------------------------------
        fig = plot_results(vv_arr, vh_arr, prob_mean, prob_std, mask_bin, hazard_label)
        st.pyplot(fig)
        plt.close(fig)

        # ---- download buttons ----------------------------------------------
        st.subheader("⬇️ Download Results")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        prob_buf = io.BytesIO()
        np.save(prob_buf, prob_mean)
        col_dl1.download_button(
            "Probability map (.npy)", prob_buf.getvalue(),
            file_name="flood_probability.npy", mime="application/octet-stream",
        )

        mask_buf = io.BytesIO()
        np.save(mask_buf, mask_bin)
        col_dl2.download_button(
            "Binary mask (.npy)", mask_buf.getvalue(),
            file_name="flood_mask.npy", mime="application/octet-stream",
        )

        fig2 = plot_results(vv_arr, vh_arr, prob_mean, prob_std, mask_bin, hazard_label)
        img_buf = io.BytesIO()
        fig2.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig2)
        col_dl3.download_button(
            "Summary figure (.png)", img_buf.getvalue(),
            file_name="casa_net_result.png", mime="image/png",
        )


def page_demo():
    st.header("🧪 Demo – Synthetic SAR Scene")
    st.markdown(
        """
        No data? This page generates a **synthetic SAR scene** with simulated flood areas
        and runs the model on it so you can explore the interface.
        """
    )

    seed = st.sidebar.slider("Random seed", 0, 100, 42)
    h = st.sidebar.slider("Scene height (px)", 64, 512, 256, step=64)
    w = st.sidebar.slider("Scene width (px)", 64, 512, 256, step=64)
    threshold = st.sidebar.slider("Threshold", 0.1, 0.9, DEFAULT_THRESHOLD, 0.05)

    if st.button("▶️ Generate & Infer", type="primary"):
        rng = np.random.default_rng(seed)

        # Simulate Sentinel-1 backscatter values (dB → linear scale)
        vv_arr = rng.normal(-10, 3, (h, w)).astype(np.float32)
        vh_arr = rng.normal(-17, 3, (h, w)).astype(np.float32)

        # Introduce a flood rectangle (lower backscatter = smooth water)
        r0, r1 = int(0.3 * h), int(0.6 * h)
        c0, c1 = int(0.25 * w), int(0.75 * w)
        vv_arr[r0:r1, c0:c1] -= 8
        vh_arr[r0:r1, c0:c1] -= 8

        with st.spinner("Running inference …"):
            t_vv, t_vh = preprocess_sar(vv_arr, vh_arr)
            terrain_np = make_dummy_terrain(h, w)
            t_terrain = torch.tensor(terrain_np[None, ...], dtype=torch.float32)
            model = load_model(None)
            prob_mean, prob_std, mask_bin = run_inference(
                model, t_vv, t_vh, t_terrain, threshold=threshold
            )

        fig = plot_results(vv_arr, vh_arr, prob_mean, prob_std, mask_bin, "Flood (synthetic)")
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "ℹ️ This demo uses **randomly initialised weights** – the flood detection "
            "above is not meaningful. Upload a trained `.pth` checkpoint on the "
            "Inference page for real results."
        )


def page_about():
    st.header("ℹ️ About CASA-Net")
    st.markdown(
        """
        ## Cross-polarization Attention SAR Asymmetric Network (CASA-Net)

        CASA-Net is a deep-learning segmentation model designed for flood mapping
        from dual-polarisation (VV + VH) Sentinel-1 SAR imagery.

        ### Architecture Overview

        | Component | Description |
        |-----------|-------------|
        | **VV Encoder** | ResNet34 (pre-trained ImageNet, adapted for 1-channel SAR) |
        | **VH Encoder** | MobileNetV3-Small (pre-trained, adapted for 1-channel SAR) |
        | **CPAG** | Cross-Polarisation Attention Gate – spatial attention between polarisations |
        | **Terrain Branch** | FiLM conditioning from slope / TWI / HAND / JRC water layers |
        | **Decoder** | U-Net decoder with FiLM modulation at each stage |
        | **Head** | MC Dropout + 1×1 Conv + Sigmoid for uncertainty-aware prediction |

        ### Inference
        - **Monte Carlo Dropout** (10 forward passes with dropout active) provides
          pixel-wise epistemic uncertainty estimates.
        - **Threshold** controls the probability cutoff for the binary flood mask.

        ### Multi-Hazard Support
        The model supports separate branches for **flood**, **erosion**, and **landslide**
        predictions (controlled by the `MULTI_HAZARD_MODE` environment variable).

        ### Environment Variables

        | Variable | Default | Description |
        |----------|---------|-------------|
        | `CASA_FORCE_CPU` | `0` | Set to `1` to force CPU inference |
        | `MULTI_HAZARD_MODE` | `flood` | Comma-separated list of active hazards |

        ### Study Area
        The model was trained on Sentinel-1 SAR data covering **flood-prone regions
        in Bangladesh**, including Sylhet, Sunamganj, and surrounding haor wetlands.

        ### Citation
        If you use this model, please cite the CASA-Net paper (see `README.md`).
        """
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="CASA-Net | SAR Flood Segmentation",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Flag_of_Bangladesh.svg/320px-Flag_of_Bangladesh.svg.png",
        width=120,
    )
    st.sidebar.title("CASA-Net")
    st.sidebar.caption("SAR Flood Segmentation · Bangladesh")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🌊 Flood Inference", "🧪 Demo", "ℹ️ About"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: **{DEVICE}**")
    st.sidebar.caption(f"Active hazards: **{', '.join(MULTI_HAZARD_MODE)}**")

    if page == "🌊 Flood Inference":
        page_inference()
    elif page == "🧪 Demo":
        page_demo()
    else:
        page_about()


if __name__ == "__main__":
    main()
