"""UNetEx definition and weight loading for the inference service.

The architecture is a verbatim copy of the inline definition in ``app.py`` so
that the service has no dependency on Streamlit or on the ``deepcfd`` package.
Keep the two in sync if the architecture ever changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

log = logging.getLogger("swiftcfd.model")

# ── Architecture (identical to app.py) ────────────────────────────


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
        _in = in_channels * 2 if i == 0 else in_channels
        _out = in_channels
        _bn = bn
        _act = activation
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
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
            x = _x
            tensors = _tensors[:]
            indices = _indices[:]
            sizes = _sizes[:]
            for decoder in _decoder:
                tensor = tensors.pop()
                size = sizes.pop()
                ind = indices.pop()
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


# ── Loading ────────────────────────────────────────────────────────

HF_REPO = "vamsigudipati/deepcfd-model"
HF_FILENAME = "mymodel_v2.pt"
FILTERS = [8, 16, 32, 32]
KERNEL_SIZE = 5
FIXTURE_SEED = 0

# Keys that the training scripts store alongside the weights (see app.py).
_METADATA_KEYS = ("filters", "kernel_size", "input_shape", "architecture")


@dataclass(frozen=True)
class LoadedModel:
    model: UNetEx
    version: str
    source: str
    n_parameters: int


def build_model() -> UNetEx:
    model = UNetEx(3, 3, filters=FILTERS, kernel_size=KERNEL_SIZE,
                   batch_norm=False, weight_norm=False)
    model.eval()
    return model


def _load_state_dict(path: str, model: UNetEx) -> None:
    state_dict = torch.load(path, map_location="cpu")
    for key in _METADATA_KEYS:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict)


def load_model(source: str | None = None) -> LoadedModel:
    """Load the surrogate.

    ``source`` (defaults to ``$SWIFTCFD_WEIGHTS``, then ``"hf"``) is one of:

    * ``"hf"``       – download ``mymodel_v2.pt`` from the Hugging Face Hub
    * ``"fixture"``  – deterministic seeded random weights with the *real*
                       architecture. Used by the tests and CI so the pipeline
                       can be exercised without network access. **Not** a
                       trained model.
    * a file path    – a local ``.pt`` state dict.
    """
    source = source or os.environ.get("SWIFTCFD_WEIGHTS", "hf")
    model = build_model()

    if source == "fixture":
        torch.manual_seed(FIXTURE_SEED)
        model = build_model()  # re-initialise under the seed
        version = f"fixture-seed{FIXTURE_SEED}"
        log.warning("Loaded FIXTURE weights (seeded random). Predictions are not physical.")
    elif source == "hf":
        from huggingface_hub import hf_hub_download  # imported lazily: optional at test time

        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        _load_state_dict(path, model)
        version = os.environ.get("SWIFTCFD_MODEL_VERSION", HF_FILENAME.removesuffix(".pt"))
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"SWIFTCFD_WEIGHTS={source!r} is not a file")
        _load_state_dict(source, model)
        version = os.environ.get(
            "SWIFTCFD_MODEL_VERSION", os.path.basename(source).removesuffix(".pt"))

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model loaded: version=%s source=%s params=%d", version, source, n_params)
    return LoadedModel(model=model, version=version, source=source, n_parameters=n_params)


@torch.no_grad()
def predict(model: UNetEx, x: torch.Tensor) -> torch.Tensor:
    """Run inference on a ``(3, NX, NY)`` or ``(B, 3, NX, NY)`` tensor."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return model(x)
