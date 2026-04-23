import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torch.nn.utils import weight_norm


# ══════════════════════════════════════════════════════════════════
# INLINE MODEL DEFINITION (no deepcfd package dependency)
# ═════════════════════════════════════════��════════════════════════

def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, convolution=nn.Conv2d):
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)


def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2):
    encoder = []
    for i in range(layers):
        _in = out_channels if i > 0 else in_channels
        encoder.append(create_layer(_in, out_channels, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)


def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2, final_layer=False):
    decoder = []
    for i in range(layers):
        _in  = in_channels * 2 if i == 0 else in_channels
        _out = in_channels
        _bn  = bn
        _act = activation
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn  = False
                _act = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _act, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)


def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True,
                   activation=nn.ReLU, layers=2):
    encoder = []
    for i, f in enumerate(filters):
        in_c = in_channels if i == 0 else filters[i - 1]
        encoder.append(create_encoder_block(in_c, f, kernel_size, wn, bn, activation, layers))
    return nn.Sequential(*encoder)


def create_decoder(out_channels, filters, kernel_size, wn=True, bn=True,
                   activation=nn.ReLU, layers=2):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(
                filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True)
        else:
            decoder_layer = create_decoder_block(
                filters[i], filters[i - 1], kernel_size, wn, bn, activation, layers)
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)


class UNetEx(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 layers=3, weight_norm=True, batch_norm=True, activation=nn.ReLU,
                 final_activation=None):
        super().__init__()
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size,
                                      weight_norm, batch_norm, activation, layers)
        self.decoders = nn.Sequential(*[
            create_decoder(1, filters, kernel_size, weight_norm, batch_norm, activation, layers)
            for _ in range(out_channels)
        ])

    def encode(self, x):
        tensors, indices, sizes = [], [], []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, _x, _tensors, _indices, _sizes):
        y = []
        for _decoder in self.decoders:
            x       = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes   = _sizes[:]
            for decoder in _decoder:
                tensor = tensors.pop()
                size   = sizes.pop()
                ind    = indices.pop()
                x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
                x = torch.cat([tensor, x], dim=1)
                x = decoder(x)
            y.append(x)
        return torch.cat(y, dim=1)

    def forward(self, x):
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SwiftCFD — Neural Flow Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title  { font-size: 2.6rem; font-weight: 800; color: #1E88E5; }
    .subtitle    { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .badge       { background:#e3f2fd; color:#1565C0; border-radius:6px;
                   padding:3px 10px; font-size:0.85rem; font-weight:600; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD MODEL & DATA (cached — downloaded once per session)
# ══════════════════════════════════════════════════════════════════

HF_REPO = "vamsigudipati/deepcfd-model"


@st.cache_resource(show_spinner="⏳ Loading SwiftCFD model from Hugging Face...")
def load_model():
    path = hf_hub_download(repo_id=HF_REPO, filename="mymodel_v2.pt")
    state_dict = torch.load(path, map_location="cpu")
    for key in ["filters", "kernel_size", "input_shape", "architecture"]:
        state_dict.pop(key, None)
    model = UNetEx(3, 3, filters=[8, 16, 32, 32], kernel_size=5,
                   batch_norm=False, weight_norm=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner="⏳ Loading dataset from Hugging Face...")
def load_data():
    x_path = hf_hub_download(repo_id=HF_REPO, filename="dataX.pkl")
    y_path = hf_hub_download(repo_id=HF_REPO, filename="dataY.pkl")
    x = torch.FloatTensor(pickle.load(open(x_path, "rb")))
    y = torch.FloatTensor(pickle.load(open(y_path, "rb")))
    return x, y


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def run_inference(model, x_input):
    with torch.no_grad():
        if x_input.dim() == 3:
            x_input = x_input.unsqueeze(0)
        return model(x_input).squeeze(0)


def mse(pred, gt):
    return float(torch.mean((pred - gt) ** 2))


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_results(pred, gt=None, display_mode="Prediction Only"):
    field_names = ["Ux", "Uy", "p"]
    field_units = ["m/s", "m/s", "Pa"]

    nx = pred.shape[1]
    ny = pred.shape[2]
    plot_options = {"cmap": "jet", "origin": "lower", "extent": [0, nx, 0, ny]}

    if display_mode == "Prediction Only" or gt is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("SwiftCFD — Neural Network Prediction",
                     fontsize=14, fontweight="bold")
        for i, (name, unit) in enumerate(zip(field_names, field_units)):
            pr = pred[i].numpy().T
            im = axes[i].imshow(pr, **plot_options)
            axes[i].set_title(f"{name} ({unit})", fontsize=11)
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            plt.colorbar(im, ax=axes[i], orientation="horizontal")
    else:
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("SwiftCFD — CFD Ground Truth vs SwiftCFD Prediction vs Error",
                     fontsize=13, fontweight="bold")
        for i, (name, unit) in enumerate(zip(field_names, field_units)):
            gt_f  = gt[i].numpy()
            pr_f  = pred[i].numpy()
            err   = np.abs(gt_f - pr_f)
            vmin, vmax = gt_f.min(), gt_f.max()
            emin, emax = err.min(), err.max()

            ax1 = plt.subplot(3, 3, i * 3 + 1)
            if i == 0:
                ax1.set_title("CFD Ground Truth", fontsize=13)
            im1 = ax1.imshow(np.transpose(gt_f), vmin=vmin, vmax=vmax, **plot_options)
            plt.colorbar(im1, ax=ax1, orientation="horizontal")
            ax1.set_ylabel(name, fontsize=13)

            ax2 = plt.subplot(3, 3, i * 3 + 2)
            if i == 0:
                ax2.set_title("SwiftCFD Prediction", fontsize=13)
            im2 = ax2.imshow(np.transpose(pr_f), vmin=vmin, vmax=vmax, **plot_options)
            plt.colorbar(im2, ax=ax2, orientation="horizontal")

            ax3 = plt.subplot(3, 3, i * 3 + 3)
            if i == 0:
                ax3.set_title("Error", fontsize=13)
            im3 = ax3.imshow(np.transpose(err), vmin=emin, vmax=emax, **plot_options)
            plt.colorbar(im3, ax=ax3, orientation="horizontal")

    plt.tight_layout()
    return fig


def compute_sdf(mask):
    mask   = mask.astype(float)
    d_out  = ndimage.distance_transform_edt(1 - mask)
    d_in   = ndimage.distance_transform_edt(mask)
    sdf    = d_out - d_in
    maxval = max(abs(sdf).max(), 1e-8)
    return (sdf / maxval).astype(np.float32)


def build_input_tensor(mask, ux_inlet, H, W):
    sdf = compute_sdf(mask)
    ux  = np.ones((H, W), dtype=np.float32) * ux_inlet
    uy  = np.zeros((H, W), dtype=np.float32)
    ux[mask == 1] = 0.0
    return torch.FloatTensor(np.stack([sdf, ux, uy], axis=0))


def shape_to_mask(shape_type, H, W, cx, cy, size):
    mask = np.zeros((H, W), dtype=np.uint8)
    Y, X = np.ogrid[:H, :W]
    if shape_type == "Rectangle":
        mask[max(0, cy - size // 2):min(H, cy + size // 2),
             max(0, cx - size):min(W, cx + size)] = 1
    elif shape_type == "Circle":
        mask[np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) <= size] = 1
    elif shape_type == "Diamond":
        mask[np.abs(X - cx) + np.abs(Y - cy) <= size] = 1
    elif shape_type == "Triangle":
        for y in range(H):
            for x in range(W):
                dy = abs(y - cy)
                if cx - size <= x <= cx + size and dy <= size - abs(x - cx):
                    mask[y, x] = 1
    return mask


def canvas_to_mask(canvas_data, H, W):
    if canvas_data is None:
        return np.zeros((H, W), dtype=np.uint8)
    gray = np.mean(canvas_data[:, :, :3], axis=2)
    raw  = (gray < 128).astype(np.uint8)
    img  = Image.fromarray(raw * 255).resize((W, H), Image.NEAREST)
    return (np.array(img) > 128).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════���═══════════════════

st.markdown('<div class="main-title">🌊 SwiftCFD</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Neural Network-powered Computational Fluid Dynamics — '
    'instant flow field predictions replacing hours of simulation with milliseconds.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<span class="badge">UNet Architecture</span>&nbsp;'
    '<span class="badge">981 CFD Training Samples</span>&nbsp;'
    '<span class="badge">Val MSE 0.739</span>&nbsp;'
    '<span class="badge">~50ms per prediction</span>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Load resources ─────────────────────────────────────────────────
model        = load_model()
dataX, dataY = load_data()
N, C, H, W   = dataX.shape

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Settings")
display_mode = st.sidebar.radio(
    "Display Mode",
    ["Prediction Only", "Ground Truth + Prediction + Error"],
    help="'Ground Truth' only available for Dataset tab"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Model Info**")
st.sidebar.markdown(f"""
| Property | Value |
|---|---|
| Architecture | UNet + skip connections |
| Grid size | {H} × {W} |
| Dataset | {N} samples |
| Best Val MSE | 0.739 |
| Training epochs | 213 |
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **vamsigudipati** · "
    "[Model on HF](https://huggingface.co/vamsigudipati/deepcfd-model)"
)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📂  Pick from Dataset",
    "⬆️  Upload Geometry",
    "✏️  Draw Obstacle",
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — Pick from Dataset
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Select a sample from the CFD dataset")
    col_s, col_m = st.columns([4, 1])
    with col_s:
        sample_idx = st.slider("Sample Index", 0, N - 1, 0, key="slider_sample")
    with col_m:
        st.metric("Total Samples", N)

    fig_geo, ax_geo = plt.subplots(1, 1, figsize=(9, 3))
    ax_geo.imshow(dataX[sample_idx, 0].numpy().T, cmap="gray", origin="lower")
    ax_geo.set_title(f"Sample #{sample_idx} — Input Geometry (SDF channel)", fontsize=11)
    ax_geo.set_xlabel("x")
    ax_geo.set_ylabel("y")
    plt.tight_layout()
    st.pyplot(fig_geo)
    plt.close()

    if st.button("🚀 Run SwiftCFD Prediction", key="run_dataset"):
        with st.spinner("Running inference..."):
            pred = run_inference(model, dataX[sample_idx])
            gt   = dataY[sample_idx]

        st.success("✅ Done!")

        if display_mode == "Ground Truth + Prediction + Error":
            c1, c2, c3, c4 = st.columns(4)
            # ✅ FIX: scientific notation so small values display correctly
            c1.metric("Ux MSE",    f"{mse(pred[0], gt[0]):.2e}")
            c2.metric("Uy MSE",    f"{mse(pred[1], gt[1]):.2e}")
            c3.metric("p  MSE",    f"{mse(pred[2], gt[2]):.2e}")
            c4.metric("Total MSE", f"{mse(pred, gt):.2e}")

        fig = plot_results(pred, gt, display_mode)
        st.pyplot(fig)
        st.download_button("⬇️ Download Plot", fig_to_bytes(fig),
                           f"swiftcfd_sample_{sample_idx}.png", "image/png")
        plt.close()

# ─────────────────────────────────────────────────────────────────
# TAB 2 — Upload Geometry
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload a custom geometry file")
    st.info("""
Upload a `.pkl` file containing a **numpy array of shape `(3, H, W)`**:
- **Channel 0** — Signed Distance Function (SDF) of the obstacle
- **Channel 1** — Inlet Ux velocity field
- **Channel 2** — Inlet Uy velocity field
    """)

    uploaded = st.file_uploader("Choose a .pkl file", type=["pkl"])

    if uploaded:
        try:
            data    = pickle.load(uploaded)
            x_input = torch.FloatTensor(data)

            if x_input.dim() == 3 and x_input.shape[0] == 3:
                st.success(f"✅ File loaded — shape: {tuple(x_input.shape)}")

                fig_up, axes_up = plt.subplots(1, 3, figsize=(15, 3))
                titles = ["SDF (Ch 0)", "Ux inlet (Ch 1)", "Uy inlet (Ch 2)"]
                for i in range(3):
                    im = axes_up[i].imshow(x_input[i].numpy().T, cmap="jet", origin="lower")
                    axes_up[i].set_title(titles[i])
                    plt.colorbar(im, ax=axes_up[i], orientation="horizontal")
                plt.tight_layout()
                st.pyplot(fig_up)
                plt.close()

                if st.button("🚀 Run SwiftCFD Prediction", key="run_upload"):
                    with st.spinner("Running inference..."):
                        pred = run_inference(model, x_input)
                    st.success("✅ Done!")
                    fig = plot_results(pred, display_mode="Prediction Only")
                    st.pyplot(fig)
                    st.download_button("⬇️ Download Plot", fig_to_bytes(fig),
                                       "swiftcfd_upload.png", "image/png")
                    plt.close()
            else:
                st.error(f"❌ Expected shape (3, H, W) — got {tuple(x_input.shape)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ─────────────────────────────────────────────────────────────────
# TAB 3 — Draw Obstacle
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Design your own obstacle")

    draw_mode = st.radio(
        "Drawing Mode",
        ["🔷 Shape Selector", "✏️ Freehand Draw"],
        horizontal=True
    )

    ux_inlet = st.slider("Inlet Velocity Ux (m/s)", 0.01, 0.30, 0.10, step=0.01,
                         help="Inlet x-velocity applied at the left boundary")

    st.markdown("---")

    # ── Shape Selector ─────────────────────────────────────────────
    if draw_mode == "🔷 Shape Selector":
        col_l, col_r = st.columns(2)
        with col_l:
            shape = st.selectbox("Shape", ["Rectangle", "Circle", "Diamond", "Triangle"])
            cx    = st.slider("Center X", W // 6, W - W // 6, W // 3, key="cx")
            cy    = st.slider("Center Y", H // 6, H - H // 6, H // 2, key="cy")
        with col_r:
            size  = st.slider("Size (radius / half-width)", 3, min(H, W) // 3, 10, key="sz")
            st.markdown(f"**Grid:** `{H} × {W}` cells")

        mask = shape_to_mask(shape, H, W, cx, cy, size)

        fig_prev, ax_prev = plt.subplots(1, 2, figsize=(13, 3))
        ax_prev[0].imshow(mask.T, cmap="gray", origin="lower")
        ax_prev[0].set_title("Obstacle Mask")
        ax_prev[0].set_xlabel("x")
        ax_prev[0].set_ylabel("y")
        sdf_p = compute_sdf(mask)
        im_s  = ax_prev[1].imshow(sdf_p.T, cmap="RdBu", origin="lower")
        ax_prev[1].set_title("Signed Distance Function (model input)")
        plt.colorbar(im_s, ax=ax_prev[1], orientation="horizontal")
        plt.tight_layout()
        st.pyplot(fig_prev)
        plt.close()

        if st.button("🚀 Run SwiftCFD Prediction", key="run_shape"):
            with st.spinner("Running inference..."):
                x_input = build_input_tensor(mask, ux_inlet, H, W)
                pred    = run_inference(model, x_input)
            st.success("✅ Done!")
            fig = plot_results(pred, display_mode="Prediction Only")
            st.pyplot(fig)
            st.download_button("⬇️ Download Plot", fig_to_bytes(fig),
                               f"swiftcfd_{shape.lower()}.png", "image/png")
            plt.close()

    # ── Freehand Draw ──────────────────────────────────────────────
    else:
        st.markdown("**Paint your obstacle in black below:**")

        # ✅ FIX: persist canvas mask in session_state so button click doesn't lose it
        if "canvas_mask" not in st.session_state:
            st.session_state.canvas_mask = None

        CANVAS_H = 316
        CANVAS_W = int(CANVAS_H * W / H)

        canvas_result = st_canvas(
            fill_color       = "rgba(0,0,0,1)",
            stroke_width     = 18,
            stroke_color     = "#000000",
            background_color = "#FFFFFF",
            height           = CANVAS_H,
            width            = CANVAS_W,
            drawing_mode     = "freedraw",
            key              = "canvas_draw",
        )

        # Capture mask immediately when canvas updates
        if canvas_result.image_data is not None:
            captured = canvas_to_mask(canvas_result.image_data, H, W)
            if captured.sum() > 0:
                st.session_state.canvas_mask = captured

        # Preview
        if st.session_state.canvas_mask is not None:
            mask = st.session_state.canvas_mask
            fig_fh, ax_fh = plt.subplots(1, 2, figsize=(13, 3))
            ax_fh[0].imshow(mask.T, cmap="gray", origin="lower")
            ax_fh[0].set_title("Obstacle Mask (from drawing)")
            sdf_fh = compute_sdf(mask)
            im_fh  = ax_fh[1].imshow(sdf_fh.T, cmap="RdBu", origin="lower")
            ax_fh[1].set_title("Signed Distance Function (model input)")
            plt.colorbar(im_fh, ax=ax_fh[1], orientation="horizontal")
            plt.tight_layout()
            st.pyplot(fig_fh)
            plt.close()

        col_btn, col_clr = st.columns([2, 1])
        with col_btn:
            run_btn = st.button("🚀 Run SwiftCFD Prediction", key="run_freehand")
        with col_clr:
            # ✅ FIX: Clear button properly resets saved mask
            if st.button("🗑️ Clear Drawing", key="clear_canvas"):
                st.session_state.canvas_mask = None
                st.rerun()

        if run_btn:
            if st.session_state.canvas_mask is None:
                st.warning("⚠️ Please draw an obstacle first!")
            else:
                with st.spinner("Running inference..."):
                    x_input = build_input_tensor(
                        st.session_state.canvas_mask, ux_inlet, H, W)
                    pred = run_inference(model, x_input)
                st.success("✅ Done!")
                fig = plot_results(pred, display_mode="Prediction Only")
                st.pyplot(fig)
                st.download_button("⬇️ Download Plot", fig_to_bytes(fig),
                                   "swiftcfd_freehand.png", "image/png")
                plt.close()

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.85rem;'>
🌊 SwiftCFD — UNet surrogate for Computational Fluid Dynamics &nbsp;|&nbsp;
Trained on 981 CFD samples &nbsp;|&nbsp;
⚡ ~50ms vs hours for traditional simulation
</div>
""", unsafe_allow_html=True)