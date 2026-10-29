"""Compression stage: high-dim per-ROP features -> compact latent.

The windowed input is ``(N, k, F)`` with ``F`` on the order of a few hundred
counters. Before the temporal model we compress each per-ROP feature vector to
a small ``latent_dim`` (default 32), which cuts cost and denoises.

Two interchangeable compressors share one interface (``fit`` / ``transform``):

* :class:`PCACompressor` - linear, deterministic, instant. The baseline.
* :class:`MAECompressor` - a masked autoencoder (the autofm design choice): a
  small MLP encoder/decoder trained to reconstruct randomly-masked inputs, so
  the latent captures nonlinear cross-level structure. Reconstruction error
  doubles as an anomaly score.

Both are fit on the **training** feature rows only (flattened over time), then
applied to every window, preserving the ``(N, k, latent)`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional for the PCA-only path
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _flatten_time(X: np.ndarray) -> np.ndarray:
    """(N, k, F) -> (N*k, F) for fitting a per-ROP compressor."""
    n, k, f = X.shape
    return X.reshape(n * k, f)


def _unflatten_time(flat: np.ndarray, n: int, k: int) -> np.ndarray:
    """(N*k, L) -> (N, k, L)."""
    return flat.reshape(n, k, flat.shape[1])


# --------------------------------------------------------------------------- #
# PCA baseline
# --------------------------------------------------------------------------- #
@dataclass
class PCACompressor:
    """Linear PCA compressor to ``latent_dim`` components."""

    latent_dim: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=self.latent_dim, random_state=self.seed)

    def fit(self, X_train: np.ndarray) -> "PCACompressor":
        self._pca.fit(_flatten_time(X_train))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n, k, _ = X.shape
        z = self._pca.transform(_flatten_time(X))
        return _unflatten_time(z.astype(np.float32), n, k)

    @property
    def explained_variance(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())


# --------------------------------------------------------------------------- #
# Masked Autoencoder (autofm design choice)
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class _MAENet(nn.Module):
        """Small MLP autoencoder: F -> 64 -> latent -> 64 -> F."""

        def __init__(self, in_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, in_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(self.encoder(x))


@dataclass
class MAECompressor:
    """Masked autoencoder compressor to ``latent_dim``.

    Trains on per-ROP feature rows with a fraction of inputs masked to 0, and
    reconstructs the full row. The encoder then maps any row to its latent.
    """

    latent_dim: int = 32
    mask_frac: float = 0.4
    epochs: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42
    amp: bool = True
    num_workers: int = 0
    pin_memory: bool | None = None

    def __post_init__(self) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("MAECompressor requires torch; use PCACompressor.")
        torch.manual_seed(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        self._net: "_MAENet | None" = None

    def fit(self, X_train: np.ndarray) -> "MAECompressor":
        from torch.utils.data import DataLoader, TensorDataset

        # TF32 speeds up fp32 matmuls/convs on Ampere+ GPUs (no-op elsewhere).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # copy=False: the windows are already float32, and a second copy of
        # every row is what runs a long history out of memory.
        flat = _flatten_time(X_train).astype(np.float32, copy=False)
        self._net = _MAENet(flat.shape[1], self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # AMP + pinned memory + NVTX are CUDA-only; on CPU/MPS they stay off so
        # the same code runs unchanged on a laptop.
        is_cuda = self.device.type == "cuda"
        use_amp = bool(self.amp) and is_cuda
        pin = self.pin_memory if self.pin_memory is not None else is_cuda

        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(flat)),
            batch_size=self.batch_size, shuffle=True, generator=g,
            num_workers=self.num_workers, pin_memory=pin,
            persistent_workers=(self.num_workers > 0),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def _push(t):
            if is_cuda:
                torch.cuda.nvtx.range_push(t)

        def _pop():
            if is_cuda:
                torch.cuda.nvtx.range_pop()

        n = len(flat)
        self._net.train()
        for epoch in range(self.epochs):
            _push(f"mae/epoch_{epoch}")
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=pin)
                # Random mask: zero a fraction of inputs, reconstruct the full row.
                mask = (torch.rand_like(xb) > self.mask_frac).float()
                opt.zero_grad(set_to_none=True)
                _push("mae/forward")
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=use_amp):
                    recon = self._net(xb * mask)
                    loss = loss_fn(recon, xb)
                _pop()
                _push("mae/backward")
                scaler.scale(loss).backward()
                _pop()
                _push("mae/optim")
                scaler.step(opt)
                scaler.update()
                _pop()
                total += loss.item() * len(xb)
            _pop()
            print(f"  [MAE] epoch {epoch + 1}/{self.epochs} "
                  f"recon_mse={total / n:.4f}")
        return self

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "fit() first"
        self._net.eval()
        n, k, _ = X.shape
        flat = _flatten_time(X).astype(np.float32, copy=False)
        # Encode in chunks: moving every row to the GPU at once does not fit
        # for long histories.
        z = np.empty((len(flat), self.latent_dim), dtype=np.float32)
        step = 262_144
        for i in range(0, len(flat), step):
            xb = torch.from_numpy(flat[i:i + step]).to(self.device)
            z[i:i + step] = self._net.encoder(xb).float().cpu().numpy()
        return _unflatten_time(z, n, k)


def build_compressor(kind: str, latent_dim: int = 32, **kw):
    """Factory: ``kind`` in {"pca", "mae"}."""
    if kind == "pca":
        return PCACompressor(latent_dim=latent_dim, seed=kw.get("seed", 42))
    if kind == "mae":
        return MAECompressor(latent_dim=latent_dim, **kw)
    raise ValueError(f"Unknown compressor kind: {kind!r}")
