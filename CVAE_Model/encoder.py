"""
Conditional VAE Encoder — MIF grid → deterministic condition vector

Architecture (4-stage residual, input-size-agnostic):

    Input:  (B, 5, D, H, W)           — arbitrary spatial size
      │
      ├─ InputNorm (learnable instance norm)
      ├─ Stage 1: ResConvBlock(1→16,  k=3, s=2, GN-4)   — 4 groups for 16ch
      ├─ Stage 2: ResConvBlock(16→32, k=3, s=2, GN-8)
      ├─ Stage 3: ResConvBlock(32→64, k=3, s=2, GN-8)
      ├─ Stage 4: ResConvBlock(64→128,k=3, s=2, GN-8)
      │
      ├─ AdaptiveAvgPool3d(1)         — (B, 128, 1, 1, 1)
      ├─ Flatten                      — (B, 128)
      │
      ├─ FC_out: Linear(128 → latent_dim)
      └─ LayerNorm(latent_dim)        — (B, latent_dim) = c

The MIF encoder is deterministic — it produces a single condition vector
c, not a distribution.  This is correct for a conditional VAE where the
condition (the pocket) is fully observed and the generative diversity
comes from sampling the ligand latent z ~ N(0, I).

Key design choices:
    - Deterministic output (no mu/logvar split) — the condition is observed,
      not sampled.  Diversity comes from z, not from c.
    - LayerNorm on output — forces c to zero mean / unit variance, putting
      it on the same scale as z (KL-regularised toward N(0,I)) so the
      decoder sees balanced [z, c] inputs.
    - Residual connections for stable gradient flow through 4 stages
    - AdaptiveAvgPool3d removes FC over-parameterisation & input-size coupling
    - Dynamic GroupNorm groups (min 4 channels/group) for stability in early stages
    - Correct Kaiming init (gain≈√2 for GELU, not gain=1)
    - Small FC init so output starts near zero before LayerNorm centres it
    - Learnable input normalisation
    - Optional gradient checkpointing for VRAM savings
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """All encoder hyperparameters in one place."""

    # -- Input --
    in_channels: int = 5            # MIF channels (e.g. donor, acceptor, hydrophobic, positive, negative)
    # -- Conv backbone --
    base_width: int = 16            # first conv width; doubles each stage
    num_stages: int = 4             # 4 × stride-2 → 16× spatial reduction
    kernel_size: int = 3            # 3×3×3 standard; 5×5×5 for wider context
    stride: int = 2                 # spatial downsampling per stage

    # -- Normalisation --
    norm_groups: int = 8            # requested GroupNorm groups (auto-reduced in early stages)
    min_channels_per_group: int = 4 # floor — prevents 2ch/group instability

    # -- Latent space --
    latent_dim: int = 64           #was 128, c dimensionality; 64-256 typical for molecular gen

    # -- Regularisation --
    dropout: float = 0.0            # 0.1-0.2 if overfitting; 0.0 for debug

    # -- Gradient checkpointing --
    gradient_checkpointing: bool = False  # True saves ~2× VRAM at ~30% speed cost

    # -- FC head init scale --
    fc_init_std: float = 0.01       # small init so output starts near zero


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_norm_groups(channels: int, requested: int, min_per_group: int = 4) -> int:
    """Return the largest number of GroupNorm groups that keeps ≥ min_per_group
    channels per group.

    Stage 1 with 16 channels and requested=8 would get 4 groups (16/4=4 ch/group),
    while later stages with 32+ channels keep all 8 groups.
    """
    return max(1, min(requested, channels // min_per_group))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    """Residual conv stage: Conv3D → GroupNorm → GELU + shortcut.

    The shortcut uses a 1×1×1 conv + GroupNorm whenever channels or spatial
    size change (i.e. every stage in our stride-2 setup).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        norm_groups: int,
        min_per_group: int,
        dropout: float,
    ):
        super().__init__()
        padding = kernel_size // 2

        # -- Main path --
        self.conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        gn_groups = _resolve_norm_groups(out_ch, norm_groups, min_per_group)
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -- Shortcut --
        needs_proj = (in_ch != out_ch) or (stride != 1)
        if needs_proj:
            sc_groups = _resolve_norm_groups(out_ch, norm_groups, min_per_group)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(sc_groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out + identity)  # residual before activation
        out = self.drop(out)
        return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Maps a single-channel 3D MIF grid to a deterministic condition vector.

    Input-size agnostic — AdaptiveAvgPool3d(1) removes the need to know
    spatial dimensions at construction time.

    Returns:
        c: (B, latent_dim) — deterministic condition vector, LayerNorm'd

    The LayerNorm on the output forces c to zero mean and unit variance
    per dimension, putting it on the same scale as the ligand latent z
    (which is KL-regularised toward N(0, I)).  This prevents the
    unregularised c from dominating the decoder when concatenated with z.
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = EncoderConfig()
        self.cfg = cfg

        # -- Learnable input normalisation --
        # Scales and shifts the raw MIF grid so the conv backbone
        # sees roughly zero-mean, unit-variance input regardless of
        # the MIF value range.
        self.input_norm = nn.InstanceNorm3d(
            cfg.in_channels, affine=True, eps=1e-5,
        )

        # -- Conv backbone (residual) --
        stages = []
        in_ch = cfg.in_channels
        for i in range(cfg.num_stages):
            out_ch = cfg.base_width * (2 ** i)
            stages.append(ResConvBlock(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                norm_groups=cfg.norm_groups,
                min_per_group=cfg.min_channels_per_group,
                dropout=cfg.dropout,
            ))
            in_ch = out_ch

        self.conv_stages = nn.ModuleList(stages)
        self.final_channels = in_ch  # after last stage

        # -- Global average pool + FC head --
        # Pooling to (1,1,1) means flat_dim = final_channels regardless
        # of input spatial size → ~16K params total for FC head vs ~16M before.
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Single deterministic output head (no mu/logvar split)
        self.fc_out = nn.Linear(self.final_channels, cfg.latent_dim)

        # -- LayerNorm on output --
        # Forces c to zero mean, unit variance per dimension.
        # Learnable affine (weight/bias) so the network can adapt the scale
        # downstream if needed, but starts from a balanced place.
        self.c_norm = nn.LayerNorm(cfg.latent_dim)

        # -- Weight init --
        self._init_weights()

    # ------------------------------------------------------------------ init

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # GELU gain ≈ ReLU gain ≈ √2; 'linear' gives gain=1 (too small)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Small init so fc_out starts near zero, then LayerNorm centres it
                nn.init.normal_(m.weight, 0, self.cfg.fc_init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------- forward

    def _forward_stages(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv stages, optionally with gradient checkpointing."""
        for stage in self.conv_stages:
            if self.cfg.gradient_checkpointing and self.training:
                x = cp.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: MIF grid, shape (B, 1, D, H, W) — any spatial size divisible
               by 2^num_stages (i.e. 16 for 4 stages).

        Returns:
            c: (B, latent_dim) — deterministic condition vector
        """
        # Input normalisation
        h = self.input_norm(x)

        # Conv backbone
        h = self._forward_stages(h)

        # Global average pool → (B, final_ch, 1, 1, 1) → (B, final_ch)
        h = self.gap(h)
        h = h.flatten(1)

        # FC head + LayerNorm
        c = self.fc_out(h)
        c = self.c_norm(c)

        return c

    # -------------------------------------------------------- convenience

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward(). Makes the intent explicit when used as
        a condition encoder in the full conditional VAE."""
        return self.forward(x)

    # --------------------------------------------------------- debug

    @torch.no_grad()
    def debug_forward(self, x: torch.Tensor) -> dict:
        """Run forward pass and return intermediate activations for debugging."""
        info: dict = {}

        h_raw = x
        info["input"] = {
            "shape": list(h_raw.shape),
            "range": [h_raw.min().item(), h_raw.max().item()],
            "mean": h_raw.mean().item(),
            "std": h_raw.std().item(),
        }

        h = self.input_norm(h_raw)
        info["after_input_norm"] = {
            "shape": list(h.shape),
            "range": [h.min().item(), h.max().item()],
            "mean": h.mean().item(),
            "std": h.std().item(),
        }

        for i, stage in enumerate(self.conv_stages):
            h = stage(h)
            info[f"stage_{i}"] = {
                "shape": list(h.shape),
                "range": [h.min().item(), h.max().item()],
                "mean": h.mean().item(),
                "std": h.std().item(),
                "zeros_pct": (h == 0).float().mean().item() * 100,
            }

        h_pooled = self.gap(h)
        info["after_gap"] = {
            "shape": list(h_pooled.shape),
            "range": [h_pooled.min().item(), h_pooled.max().item()],
        }

        h_flat = h_pooled.flatten(1)
        info["bottleneck"] = {
            "shape": list(h_flat.shape),
            "range": [h_flat.min().item(), h_flat.max().item()],
        }

        c_raw = self.fc_out(h_flat)
        info["fc_out (pre-norm)"] = {
            "shape": list(c_raw.shape),
            "range": [c_raw.min().item(), c_raw.max().item()],
            "mean": c_raw.mean().item(),
            "std": c_raw.std().item(),
        }

        c = self.c_norm(c_raw)
        info["c (post-LayerNorm)"] = {
            "shape": list(c.shape),
            "range": [c.min().item(), c.max().item()],
            "mean": c.mean().item(),
            "std": c.std().item(),
        }

        # Per-dimension health check
        c_std_per_dim = c.std(dim=0)
        info["c (post-LayerNorm)"]["std_per_dim_min"] = c_std_per_dim.min().item()
        info["c (post-LayerNorm)"]["std_per_dim_max"] = c_std_per_dim.max().item()
        info["c (post-LayerNorm)"]["std_per_dim_mean"] = c_std_per_dim.mean().item()

        if torch.isnan(c).any():
            info["c (post-LayerNorm)"]["nan_detected"] = True

        return info

    # --------------------------------------------------------- param count

    def param_counts(self) -> dict:
        """Return per-module parameter counts for profiling."""
        counts = {}
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters())
            counts[name] = n
        counts["total"] = sum(counts.values())
        return counts


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encoder debug test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[64, 64, 64],
                        help="D H W of input grid")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--checkpoint", action="store_true",
                        help="enable gradient checkpointing")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = EncoderConfig(
        latent_dim=args.latent_dim,
        base_width=args.base_width,
        gradient_checkpointing=args.checkpoint,
    )
    model = Encoder(cfg).to(args.device)

    # --- Forward pass ---
    dummy = torch.randn(args.batch_size, 5, *args.input_shape, device=args.device)
    c = model(dummy)

    print(f"\n{'='*60}")
    counts = model.param_counts()
    for name, n in counts.items():
        print(f"  {name:25s}  {n:>12,} params")
    print(f"{'='*60}")
    print(f"  Input:    {list(dummy.shape)}")
    print(f"  c:        {list(c.shape)}   range [{c.min():.4f}, {c.max():.4f}]  "
          f"mean={c.mean():.4f}  std={c.std():.4f}")
    print(f"{'='*60}\n")

    # --- Debug info ---
    info = model.debug_forward(dummy)
    for key, val in info.items():
        parts = [f"shape={val['shape']}", f"range=[{val['range'][0]:.4f}, {val['range'][1]:.4f}]"]
        if "mean" in val:
            parts.append(f"mean={val['mean']:.4f}")
        if "std" in val:
            parts.append(f"std={val['std']:.4f}")
        if "zeros_pct" in val:
            parts.append(f"zeros={val['zeros_pct']:.1f}%")
        if "std_per_dim_min" in val:
            parts.append(f"std_per_dim=[{val['std_per_dim_min']:.4f}, {val['std_per_dim_max']:.4f}]")
        if "nan_detected" in val:
            parts.append("⚠️ NaN DETECTED")
        print(f"  {key:25s}  " + "  ".join(parts))

    # --- Variable size test (input-size agnostic) ---
    print(f"\n--- Variable input size test ---")
    for size in [64, 96, 128]:
        x = torch.randn(1, 5, size, size, size, device=args.device)
        c2 = model(x)
        print(f"  input ({size}³) → c {list(c2.shape)}  "
              f"range=[{c2.min():.4f}, {c2.max():.4f}]  std={c2.std():.4f}")

    # --- Verify LayerNorm effect ---
    print(f"\n--- LayerNorm verification ---")
    print(f"  c mean (should be ~0):  {c.mean():.6f}")
    print(f"  c std  (should be ~1):  {c.std():.6f}")
    c_std_per_dim = c.std(dim=0)
    print(f"  c std per dim:  min={c_std_per_dim.min():.4f}  "
          f"max={c_std_per_dim.max():.4f}  mean={c_std_per_dim.mean():.4f}")

    print("\n✓ Encoder forward pass OK")
