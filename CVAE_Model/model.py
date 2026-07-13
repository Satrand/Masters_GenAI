"""
model.py — Conditional VAE for molecular generation.

Architecture:
    Training:
        MIF grid  → MIF Encoder  → c (condition, deterministic)
        Ligand    → Ligand Encoder → μ, σ → z (reparameterise)
        [z, c]    → Decoder      → predicted ligand
        Loss = reconstruction (Sinkhorn-matched) + KL (with free bits + annealing)

    Inference:
        MIF grid  → MIF Encoder → c
        z ~ N(0, I)              ← sample from prior
        [z, c]    → Decoder     → generated ligand

All data-dependent fields (input_shape, num_atom_types, max_atoms, etc.)
are REQUIRED — no defaults. Create CVaeConfig via DataConfig.to_model_config()
to ensure they stay in sync with your dataset.

Architecture fields (base_width, hidden_dim, etc.) have defaults and can be
tuned independently.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple, List

from encoder import Encoder, EncoderConfig
from ligand_encoder import LigandEncoder, LigandEncoderConfig, reparameterise
from decoder import Decoder, DecoderConfig
from loss import CVaELoss, LossConfig


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

@dataclass
class CVaeConfig:
    """Master config for the full conditional VAE.

    Data-dependent fields are REQUIRED — they have no defaults and must be
    set explicitly. Use DataConfig.to_model_config() to create a synced config
    from your dataset, or set them manually.

    Architecture fields have sensible defaults and can be tuned independently.
    """

    # ── Data-dependent: REQUIRED ──
    # These MUST match your dataset. Use DataConfig.to_model_config().
    input_shape: Tuple[int, int, int] = None   # (D, H, W) from DataConfig.expected_grid_shape
    in_channels: int = None                     # from DataConfig.num_channels (e.g. 5 for donor/acceptor/hydrophobic/positive/negative)
    num_atom_types: int = None                  # from DataConfig.num_atom_types (pad + elements + unk)
    max_atoms: int = None                       # from DataConfig.max_atoms
    padding_idx: int = None                     # from DataConfig.pad_id

    # ── Architecture: MIF Encoder ──
    base_width: int = 16                # double to 32 for more capacity
    num_stages: int = 4                 # 3 for small grids (32³), 5 for large (256³)
    mif_kernel_size: int = 3            # 3×3×3 standard; 5×5×5 for wider context
    mif_norm_groups: int = 8            # GroupNorm groups (auto-reduced for early stages)
    mif_dropout: float = 0.0            # 0.1-0.2 if MIF encoder overfits
    mif_gradient_checkpointing: bool = False  # True saves ~2× VRAM at ~30% speed cost

    # ── Architecture: shared ──
    latent_dim: int = 128               # z dimensionality (ligand latent)
    cond_dim: int = 64                  # NEW — c dimensionality (MIF condition)

    # ── Architecture: Ligand Encoder ──
    atom_type_emb_dim: int = 32
    use_edge_conv: bool = True          # True = EdgeConv; False = plain PointNet (legacy)
    edge_conv_dims: List[int] = field(default_factory=lambda: [128, 256])
    num_neighbors: int = 8              # k-NN neighbours for EdgeConv
    edge_aggregation: str = "max"       # "max" or "mean"
    dynamic_graph: bool = True          # True = DGCNN (k-NN in feature space after layer 0)
    ligand_pool_mode: str = "maxavg"    # "maxavg" or "max"
    head_dims: List[int] = field(default_factory=lambda: [256])
    logvar_clamp: Tuple[float, float] = (-10.0, 2.0)
    ligand_dropout: float = 0.0

    # ── Architecture: Decoder ──
    hidden_dim: int = 256
    slot_dim: int = 64
    per_atom_dims: List[int] = field(default_factory=lambda: [256, 256])
    coord_head_dims: List[int] = field(default_factory=lambda: [128, 64])
    type_head_dims: List[int] = field(default_factory=lambda: [64])
    exist_head_dims: List[int] = field(default_factory=lambda: [64])
    coord_activation: str = "none"      # "tanh" for [-1,1] coords, "none" for unbounded
    exist_prior: float = 0.5            # prior probability that a slot is a real atom
    decoder_dropout: float = 0.0

    # ── Loss: reconstruction ──
    lambda_coord: float = 1.0           # increase if coordinates are poor
    lambda_type: float = 1.0            # increase if atom types are wrong
    lambda_exist: float = 1.0           # increase if atom count is wrong
    coord_loss_type: str = "smooth_l1"  # "smooth_l1" or "mse"
    coord_cost_weight: float = 1.0      # matching cost weight for coordinates
    type_cost_weight: float = 1.0       # matching cost weight for types
    type_label_smoothing: float = 0.0   # 0.1 for mild regularisation
    max_coord_range: float = 2.0        # L1 distance range for cost normalisation (2.0 for [-1,1])
    exist_pos_weight: str = "auto"      # "auto" or fixed float for BCE pos_weight

    # ── Loss: KL ──
    lambda_kl: float = 0.001            # start low, anneal up
    free_bits_per_dim: float = 0.1      # per-dimension KL floor (0.0 = disabled)
    kl_schedule: str = "linear"         # "linear", "sigmoid", "cyclical"
    kl_anneal_steps: int = 10000        # ramp KL from 0 → lambda_kl over this many steps
    kl_cyclical_cycles: int = 4         # cycles for cyclical schedule
    kl_cyclical_prop: float = 0.5       # proportion of each cycle at full weight

    # ── Loss: matching ──
    use_sinkhorn: bool = True           # True = Sinkhorn (train), False = Hungarian always
    sinkhorn_iters: int = 20            # Sinkhorn iterations
    sinkhorn_temp: float = 1.0          # Sinkhorn temperature (lower = sharper)

    def __post_init__(self):
        """Validate that all required data-dependent fields are set."""
        required = {
            "input_shape": self.input_shape,
            "in_channels": self.in_channels,
            "num_atom_types": self.num_atom_types,
            "max_atoms": self.max_atoms,
            "padding_idx": self.padding_idx,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"CVaeConfig: missing required data-dependent fields: {missing}.\n"
                f"Use DataConfig.to_model_config() to create a synced config, "
                f"or set these fields manually."
            )

        # Shape validation
        if len(self.input_shape) != 3:
            raise ValueError(
                f"CVaeConfig: input_shape must be (D, H, W), got {self.input_shape}"
            )
        if self.num_atom_types < 2:
            raise ValueError(
                f"CVaeConfig: num_atom_types={self.num_atom_types} must be >= 2 "
                f"(pad + at least 1 element)"
            )
        if self.max_atoms <= 0:
            raise ValueError(
                f"CVaeConfig: max_atoms={self.max_atoms} must be > 0"
            )
        if self.padding_idx < 0 or self.padding_idx >= self.num_atom_types:
            raise ValueError(
                f"CVaeConfig: padding_idx={self.padding_idx} must be in "
                f"[0, {self.num_atom_types})"
            )

        # Coord activation validation
        if self.coord_activation not in ("tanh", "none"):
            raise ValueError(
                f"CVaeConfig: coord_activation must be 'tanh' or 'none', "
                f"got '{self.coord_activation}'"
            )

        # KL schedule validation
        if self.kl_schedule not in ("linear", "sigmoid", "cyclical"):
            raise ValueError(
                f"CVaeConfig: kl_schedule must be 'linear', 'sigmoid', or 'cyclical', "
                f"got '{self.kl_schedule}'"
            )

        # Edge aggregation validation
        if self.edge_aggregation not in ("max", "mean"):
            raise ValueError(
                f"CVaeConfig: edge_aggregation must be 'max' or 'mean', "
                f"got '{self.edge_aggregation}'"
            )

        # Pool mode validation
        if self.ligand_pool_mode not in ("max", "maxavg"):
            raise ValueError(
                f"CVaeConfig: ligand_pool_mode must be 'max' or 'maxavg', "
                f"got '{self.ligand_pool_mode}'"
            )


# ──────────────────────────────────────────────
#  Sub-config builders
# ──────────────────────────────────────────────

def _build_mif_config(cfg: CVaeConfig) -> EncoderConfig:
    """Build MIF encoder config from master config.

    Note: input_shape is NOT passed — the MIF encoder is input-size-agnostic
    (uses AdaptiveAvgPool3d). The input_shape is validated in CVaeConfig
    to ensure it's divisible by 2^num_stages, but the encoder doesn't need it.
    """
    return EncoderConfig(
        in_channels=cfg.in_channels,
        base_width=cfg.base_width,
        num_stages=cfg.num_stages,
        kernel_size=cfg.mif_kernel_size,
        norm_groups=cfg.mif_norm_groups,
        latent_dim=cfg.cond_dim, #mif encoder dim
        dropout=cfg.mif_dropout,
        gradient_checkpointing=cfg.mif_gradient_checkpointing,
    )


def _build_ligand_config(cfg: CVaeConfig) -> LigandEncoderConfig:
    """Build ligand encoder config from master config."""
    return LigandEncoderConfig(
        atom_type_emb_dim=cfg.atom_type_emb_dim,
        num_atom_types=cfg.num_atom_types,
        padding_idx=cfg.padding_idx,
        use_edge_conv=cfg.use_edge_conv,
        edge_conv_dims=list(cfg.edge_conv_dims),
        edge_conv_k=cfg.num_neighbors,          # was num_neighbors
        edge_conv_aggr=cfg.edge_aggregation,    # was edge_aggregation
        dynamic_graph=cfg.dynamic_graph,
        pool_mode=cfg.ligand_pool_mode,
        head_dims=list(cfg.head_dims),
        latent_dim=cfg.latent_dim,
        logvar_clamp=tuple(cfg.logvar_clamp) if cfg.logvar_clamp else (-10.0, 2.0),
        dropout=cfg.ligand_dropout,
        max_atoms=cfg.max_atoms,
    )

def _build_decoder_config(cfg: CVaeConfig) -> DecoderConfig:
    """Build decoder config from master config."""
    return DecoderConfig(
        latent_dim=cfg.latent_dim,
        cond_dim=cfg.cond_dim,
        max_atoms=cfg.max_atoms,
        num_atom_types=cfg.num_atom_types,
        padding_idx=cfg.padding_idx,
        hidden_dim=cfg.hidden_dim,
        slot_dim=cfg.slot_dim,
        per_atom_dims=list(cfg.per_atom_dims),
        coord_head_dims=list(cfg.coord_head_dims),
        type_head_dims=list(cfg.type_head_dims),
        exist_head_dims=list(cfg.exist_head_dims),
        coord_activation=cfg.coord_activation,
        exist_prior=cfg.exist_prior,
        dropout=cfg.decoder_dropout,
    )


def _build_loss_config(cfg: CVaeConfig) -> LossConfig:
    """Build loss config from master config."""
    return LossConfig(
        lambda_coord=cfg.lambda_coord,
        lambda_type=cfg.lambda_type,
        lambda_exist=cfg.lambda_exist,
        lambda_kl=cfg.lambda_kl,
        coord_cost_weight=cfg.coord_cost_weight,
        type_cost_weight=cfg.type_cost_weight,
        max_coord_range=cfg.max_coord_range,
        coord_loss_type=cfg.coord_loss_type,
        type_label_smoothing=cfg.type_label_smoothing,
        exist_pos_weight=cfg.exist_pos_weight,
        free_bits_per_dim=cfg.free_bits_per_dim,
        kl_schedule=cfg.kl_schedule,
        kl_anneal_steps=cfg.kl_anneal_steps,
        kl_cyclical_cycles=cfg.kl_cyclical_cycles,
        kl_cyclical_prop=cfg.kl_cyclical_prop,
        use_sinkhorn=cfg.use_sinkhorn,
        sinkhorn_iters=cfg.sinkhorn_iters,
        sinkhorn_temp=cfg.sinkhorn_temp,
    )


# ──────────────────────────────────────────────
#  Conditional VAE
# ──────────────────────────────────────────────

class ConditionalVAE(nn.Module):
    """
    Full conditional VAE for MIF → Ligand generation.

    Usage:
        # From DataConfig (recommended):
        data_cfg = DataConfig()
        model_cfg = data_cfg.to_model_config()
        model = ConditionalVAE(model_cfg)

        Training:
            output = model(mif_grid, atom_coords, atom_types, atom_mask, step=global_step)
            loss = output["total_loss"]
            loss.backward()

        Inference:
            result = model.generate(mif_grid, num_samples=5)
            coords, type_probs, exist_probs = result
    """

    def __init__(self, config: CVaeConfig):
        super().__init__()
        self.cfg = config

        # --- Build sub-modules ---
        self.mif_encoder = Encoder(_build_mif_config(self.cfg))
        self.ligand_encoder = LigandEncoder(_build_ligand_config(self.cfg))
        self.decoder = Decoder(_build_decoder_config(self.cfg))
        self.loss_fn = CVaELoss(
            _build_loss_config(self.cfg),
            num_atom_types=self.cfg.num_atom_types,
        )

        # --- Track training step for KL annealing ---
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # --- Input shape divisibility check ---
        divisor = 2 ** self.cfg.num_stages
        for dim in self.cfg.input_shape:
            if dim % divisor != 0:
                import warnings
                warnings.warn(
                    f"CVaeConfig: input_shape dimension {dim} is not divisible by "
                    f"2^{self.cfg.num_stages}={divisor}. The MIF encoder applies "
                    f"{self.cfg.num_stages} stride-2 convolutions, so spatial dims "
                    f"should be multiples of {divisor}."
                )

    # ---- training forward ----

    def forward(self, mif_grid, atom_coords, atom_types, atom_mask, step=None):
        """
        Full forward pass for training.

        Args:
            mif_grid:    (B, 1, D, H, W)  float32
            atom_coords: (B, N, 3)         float32
            atom_types:  (B, N)            long
            atom_mask:   (B, N)            bool
            step:        int or None       training step (for KL annealing).
                         If None, uses internal global_step counter.

        Returns:
            dict with:
                total_loss:       scalar (for .backward())
                loss_dict:        component breakdown (for logging)
                pred_coords:      (B, N, 3)   predicted coordinates
                pred_type_logits: (B, N, K)   predicted type logits
                pred_exist_logits:(B, N, 1)   predicted existence logits
                mu:               (B, z_dim)   ligand posterior mean
                logvar:           (B, z_dim)   ligand posterior log-variance
                z:                (B, z_dim)   sampled latent
                c:                (B, z_dim)   MIF condition vector (LayerNorm'd)
        """
        if step is None:
            step = self.global_step.item()

        # --- Encode MIF → condition c (deterministic) ---
        c = self.mif_encoder(mif_grid)                    # (B, latent_dim)

        # --- Encode ligand → z (with reparameterisation) ---
        z, mu, logvar = self.ligand_encoder.sample_z(
            atom_coords, atom_types, atom_mask
        )
        # z:      (B, latent_dim)
        # mu:     (B, latent_dim)
        # logvar: (B, latent_dim)

        # --- Decode [z, c] → ligand ---
        pred_coords, pred_type_logits, pred_exist_logits = self.decoder(z, c)

        # --- Compute loss ---
        total_loss, loss_dict = self.loss_fn(
            pred_coords=pred_coords,
            pred_type_logits=pred_type_logits,
            pred_exist_logits=pred_exist_logits,
            target_coords=atom_coords,
            target_types=atom_types,
            target_mask=atom_mask,
            mu=mu,
            logvar=logvar,
            step=step,
        )

        return {
            "total_loss": total_loss,
            "loss_dict": loss_dict,
            "pred_coords": pred_coords,
            "pred_type_logits": pred_type_logits,
            "pred_exist_logits": pred_exist_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "c": c,
        }

    # ---- inference: generate from MIF ----

    def generate(self, mif_grid, num_samples=1, temperature=1.0):
        """
        Generate ligand(s) conditioned on a MIF grid.

        Args:
            mif_grid:    (B, 1, D, H, W)  or (1, D, H, W)
            num_samples: int  — number of ligand samples per MIF
            temperature: float — sampling temperature.
                         Controls diversity: >1 = more diverse, <1 = more peaked.
                         Applied as sqrt(temperature) to z and 1/temperature to
                         type/existence softmax-sigmoid.

        Returns:
            coords:      (B*num_samples, N, 3)   float32 — in [-1, 1] if tanh
            type_probs:  (B*num_samples, N, K)   float32
            exist_probs: (B*num_samples, N)      float32
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # Handle unbatched input
            if mif_grid.dim() == 4:
                mif_grid = mif_grid.unsqueeze(0)

            B = mif_grid.shape[0]
            z_dim = self.cfg.latent_dim

            # Encode MIF → condition (deterministic)
            c = self.mif_encoder(mif_grid)                    # (B, latent_dim)

            # Expand condition for multiple samples
            if num_samples > 1:
                c = c.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

            # Sample z from prior N(0, I), scaled by sqrt(temperature)
            # z ~ N(0, temperature) gives variance = temperature
            z = torch.randn(B * num_samples, z_dim, device=mif_grid.device)
            if temperature != 1.0:
                z = z * math.sqrt(temperature)

            # Decode
            coords, type_logits, exist_logits = self.decoder(z, c)

            # Apply temperature to type and existence outputs
            if temperature != 1.0:
                type_probs = F.softmax(type_logits / temperature, dim=-1)
                exist_probs = torch.sigmoid(exist_logits.squeeze(-1) / temperature)
            else:
                type_probs = F.softmax(type_logits, dim=-1)
                exist_probs = torch.sigmoid(exist_logits.squeeze(-1))

        if was_training:
            self.train()

        return coords, type_probs, exist_probs

    # ---- inference: reconstruct (encode ligand then decode) ----

    def reconstruct(self, mif_grid, atom_coords, atom_types, atom_mask,
                    deterministic=True):
        """
        Encode a real ligand and reconstruct it (useful for validation).

        Args:
            mif_grid:       (B, 1, D, H, W)
            atom_coords:    (B, N, 3)
            atom_types:     (B, N)
            atom_mask:      (B, N)
            deterministic:  bool — if True, use mu (no sampling).
                            If False, sample z ~ N(mu, sigma).

        Returns:
            dict with pred_coords, pred_type_logits, pred_exist_logits,
                  mu, logvar, z, c
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            c = self.mif_encoder(mif_grid)                   # (B, latent_dim)

            if deterministic:
                # Use mu for reconstruction quality
                mu, logvar = self.ligand_encoder(atom_coords, atom_types, atom_mask)
                z = mu
            else:
                # Sample from posterior
                z, mu, logvar = self.ligand_encoder.sample_z(
                    atom_coords, atom_types, atom_mask, deterministic=False
                )

            pred_coords, pred_type_logits, pred_exist_logits = self.decoder(z, c)

        if was_training:
            self.train()

        return {
            "pred_coords": pred_coords,
            "pred_type_logits": pred_type_logits,
            "pred_exist_logits": pred_exist_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "c": c,
        }

    # ---- inference: sample from prior (no MIF) ----

    def sample_from_prior(self, batch_size, device="cpu", temperature=1.0):
        """
        Unconditional generation: sample z ~ N(0, I) with no pocket condition.

        This is mostly for debugging — the decoder expects a meaningful
        condition c, so unconditional samples may not be chemically sensible.

        Args:
            batch_size:  int
            device:      str or torch.device
            temperature: float — sampling temperature

        Returns:
            coords:      (batch_size, N, 3)
            type_probs:  (batch_size, N, K)
            exist_probs: (batch_size, N)
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            z = torch.randn(batch_size, self.cfg.latent_dim, device=device)
            if temperature != 1.0:
                z = z * math.sqrt(temperature)

            # Zero condition vector (centre of LayerNorm output)
            c = torch.zeros(batch_size, self.cfg.cond_dim, device=device)

            coords, type_logits, exist_logits = self.decoder(z, c)

            if temperature != 1.0:
                type_probs = F.softmax(type_logits / temperature, dim=-1)
                exist_probs = torch.sigmoid(exist_logits.squeeze(-1) / temperature)
            else:
                type_probs = F.softmax(type_logits, dim=-1)
                exist_probs = torch.sigmoid(exist_logits.squeeze(-1))

        if was_training:
            self.train()

        return coords, type_probs, exist_probs

    # ---- step counter ----

    def increment_step(self):
        """Call after each training step to update the KL annealing schedule."""
        self.global_step.add_(1)

    # ---- parameter count ----

    def count_parameters(self):
        """Count parameters per sub-module."""
        counts = {}
        for name, module in [
            ("mif_encoder", self.mif_encoder),
            ("ligand_encoder", self.ligand_encoder),
            ("decoder", self.decoder),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        counts["total"] = {
            "total": sum(c["total"] for c in counts.values()),
            "trainable": sum(c["trainable"] for c in counts.values()),
        }
        return counts

    # ---- debug: full forward with per-module logging ----

    def debug_forward(self, mif_grid, atom_coords, atom_types, atom_mask, step=0):
        """
        Full forward pass with detailed per-module logging.
        Use this to diagnose NaN, value collapse, or scale issues.
        """
        print("\n" + "=" * 60)
        print("  ConditionalVAE Debug Forward")
        print("=" * 60)

        # --- MIF Encoder (deterministic) ---
        print("\n--- MIF Encoder ---")
        c = self.mif_encoder(mif_grid)
        print(f"  c shape={c.shape}  range=[{c.min():.4f}, {c.max():.4f}]  "
              f"mean={c.mean():.4f}  std={c.std():.4f}")
        c_std_per_dim = c.std(dim=0)
        print(f"  c std per dim:  min={c_std_per_dim.min():.4f}  "
              f"max={c_std_per_dim.max():.4f}  mean={c_std_per_dim.mean():.4f}")
        if torch.isnan(c).any():
            print("  ⚠️  NaN in condition c!")

        # --- Ligand Encoder ---
        print("\n--- Ligand Encoder ---")
        mu, logvar = self.ligand_encoder(atom_coords, atom_types, atom_mask)
        z = reparameterise(mu, logvar)                      # standalone function
        print(f"  mu     shape={mu.shape}  range=[{mu.min():.4f}, {mu.max():.4f}]  "
              f"mean={mu.mean():.4f}  std={mu.std():.4f}")
        print(f"  logvar shape={logvar.shape}  range=[{logvar.min():.4f}, {logvar.max():.4f}]  "
              f"mean={logvar.mean():.4f}")
        print(f"  z      shape={z.shape}  range=[{z.min():.4f}, {z.max():.4f}]  "
              f"mean={z.mean():.4f}  std={z.std():.4f}")

        # Latent space health checks
        z_std_per_dim = z.std(dim=0)
        print(f"  z std per dim:  min={z_std_per_dim.min():.4f}  "
              f"max={z_std_per_dim.max():.4f}  mean={z_std_per_dim.mean():.4f}")
        if (z_std_per_dim < 0.01).any():
            n_dead = (z_std_per_dim < 0.01).sum().item()
            print(f"  ⚠️  {n_dead}/{z_std_per_dim.shape[0]} latent dims appear dead (std < 0.01)")
        if torch.isnan(z).any():
            print("  ⚠️  NaN in latent z!")

        # KL per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        print(f"  KL per dim:  min={kl_per_dim.min():.4f}  "
              f"max={kl_per_dim.max():.4f}  mean={kl_per_dim.mean():.4f}  "
              f"total={kl_per_dim.sum(dim=-1).mean():.4f}")
        n_active = (kl_per_dim > self.loss_fn.cfg.free_bits_per_dim).sum(dim=-1).float().mean()
        print(f"  Active dims (KL > free_bits): {n_active:.1f}/{kl_per_dim.shape[-1]}")

        # Scale comparison (c vs z)
        print(f"\n--- Scale check (c vs z) ---")
        print(f"  c mean abs: {c.abs().mean():.4f}   std: {c.std():.4f}")
        print(f"  z mean abs: {z.abs().mean():.4f}   std: {z.std():.4f}")
        scale_ratio = c.std() / (z.std() + 1e-8)
        if scale_ratio > 3.0 or scale_ratio < 0.33:
            print(f"  ⚠️  Scale mismatch: c/z std ratio = {scale_ratio:.2f} "
                  f"(ideal ≈ 1.0 after LayerNorm on c)")
        else:
            print(f"  ✓ c/z std ratio = {scale_ratio:.2f} — balanced")

        # --- Decoder ---
        print("\n--- Decoder ---")
        pred_coords, pred_type_logits, pred_exist_logits = self.decoder(z, c)
        exist_probs = torch.sigmoid(pred_exist_logits.squeeze(-1))
        print(f"  coords       shape={pred_coords.shape}  "
              f"range=[{pred_coords.min():.4f}, {pred_coords.max():.4f}]")
        print(f"  type_logits   shape={pred_type_logits.shape}  "
              f"range=[{pred_type_logits.min():.4f}, {pred_type_logits.max():.4f}]")
        print(f"  exist_probs   shape={exist_probs.shape}  "
              f"mean={exist_probs.mean():.4f}")
        if self.cfg.coord_activation == "tanh":
            coords_in_range = (pred_coords.abs() <= 1.0 + 1e-6).all()
            print(f"  coords in [-1,1]: {coords_in_range}")

        # --- Loss ---
        print("\n--- Loss ---")
        total_loss, loss_dict = self.loss_fn(
            pred_coords, pred_type_logits, pred_exist_logits,
            atom_coords, atom_types, atom_mask,
            mu, logvar, step=step,
        )
        for k, v in loss_dict.items():
            if isinstance(v, float):
                print(f"  {k:20s} = {v:.6f}")
            else:
                print(f"  {k:20s} = {v}")

        # --- Matching quality ---
        print("\n--- Matching (Hungarian, sample 0) ---")
        match_result = self.loss_fn.debug_match(
            pred_coords, pred_type_logits, pred_exist_logits,
            atom_coords, atom_types, atom_mask,
            sample_idx=0,
        )
        print(f"  n_real={match_result['n_real']}  n_matched={match_result['n_matched']}")
        print(f"  type_accuracy={match_result['type_accuracy']:.3f}")
        print(f"  mean_coord_dist={match_result['mean_coord_dist']:.4f}")
        print(f"  matched_exist_prob={match_result['matched_exist_prob_mean']:.3f}")
        print(f"  unmatched_exist_prob={match_result['unmatched_exist_prob_mean']:.3f}")

        print("\n" + "=" * 60)
        return total_loss, loss_dict

    # ---- save / load ----

    def save(self, path):
        """Save model checkpoint."""
        config_dict = {}
        for field_name in self.cfg.__dataclass_fields__:
            val = getattr(self.cfg, field_name)
            # Convert tuples to lists for JSON-safe serialization
            if isinstance(val, tuple):
                val = list(val)
            config_dict[field_name] = val

        torch.save({
            "model_state_dict": self.state_dict(),
            "global_step": self.global_step.item(),
            "config_dict": config_dict,
        }, path)
        print(f"Saved checkpoint to {path} (step {self.global_step.item()})")

    @classmethod
    def load(cls, path, device="cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        # Convert lists back to tuples where the config expects tuples
        config_dict = checkpoint["config_dict"].copy()
        tuple_fields = {"input_shape", "logvar_clamp"}
        for field_name in tuple_fields:
            if field_name in config_dict and isinstance(config_dict[field_name], list):
                config_dict[field_name] = tuple(config_dict[field_name])

        cfg = CVaeConfig(**config_dict)
        model = cls(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.global_step.fill_(checkpoint["global_step"])
        print(f"Loaded checkpoint from {path} (step {checkpoint['global_step']})")
        return model


# ──────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ConditionalVAE")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--max-atoms", type=int, default=88)
    parser.add_argument("--num-atom-types", type=int, default=13)
    parser.add_argument("--padding-idx", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = CVaeConfig(
        input_shape=tuple(args.input_shape),
        in_channels=5,
        num_atom_types=args.num_atom_types,
        max_atoms=args.max_atoms,
        padding_idx=args.padding_idx,
        latent_dim=args.latent_dim,
    )

    model = ConditionalVAE(cfg)

    # --- Parameter count ---
    counts = model.count_parameters()
    print("=== Parameter Count ===")
    for name, c in counts.items():
        print(f"  {name:15s}  total={c['total']:>10,}  trainable={c['trainable']:>10,}")

    # --- Dummy inputs ---
    B = args.batch_size
    N = args.max_atoms
    D, H, W = args.input_shape

    mif_grid = torch.randn(B, cfg.in_channels, D, H, W)
    atom_coords = torch.randn(B, N, 3)
    atom_types = torch.zeros(B, N, dtype=torch.long)
    atom_mask = torch.zeros(B, N, dtype=torch.bool)

    # Random real atoms
    n_real = [20, 35]
    for i, n in enumerate(n_real):
        atom_types[i, :n] = torch.randint(1, args.num_atom_types - 1, (n,))
        atom_mask[i, :n] = True

    # --- Training forward ---
    print("\n=== Training Forward ===")
    model.train()
    output = model(mif_grid, atom_coords, atom_types, atom_mask, step=0)

    print(f"  total_loss   = {output['total_loss'].item():.4f}")
    for k, v in output["loss_dict"].items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.4f}")
        else:
            print(f"  {k:20s} = {v}")
    print(f"  pred_coords  shape={output['pred_coords'].shape}")
    print(f"  pred_types   shape={output['pred_type_logits'].shape}")
    print(f"  pred_exist   shape={output['pred_exist_logits'].shape}")
    print(f"  mu           shape={output['mu'].shape}")
    print(f"  z            shape={output['z'].shape}")
    print(f"  c            shape={output['c'].shape}")

    # --- Inference: generate ---
    print("\n=== Generate (inference) ===")
    with torch.no_grad():
        coords, type_probs, exist_probs = model.generate(
            mif_grid, num_samples=3, temperature=1.0
        )
    print(f"  coords      shape={coords.shape}   (B*num_samples={B*3}, N={N})")
    print(f"  type_probs  shape={type_probs.shape}")
    print(f"  exist_probs shape={exist_probs.shape}")
    print(f"  exist mean  = {exist_probs.mean():.4f}")

    # --- Inference: reconstruct ---
    print("\n=== Reconstruct (validation) ===")
    recon = model.reconstruct(mif_grid, atom_coords, atom_types, atom_mask)
    print(f"  pred_coords  shape={recon['pred_coords'].shape}")
    print(f"  pred_types   shape={recon['pred_type_logits'].shape}")
    print(f"  Using deterministic z (mu): {recon['z'].allclose(recon['mu'])}")

    # --- Inference: sample from prior ---
    print("\n=== Sample from Prior (unconditional) ===")
    coords_prior, type_probs_prior, exist_probs_prior = model.sample_from_prior(
        batch_size=2, device="cpu"
    )
    print(f"  coords      shape={coords_prior.shape}")
    print(f"  type_probs  shape={type_probs_prior.shape}")
    print(f"  exist_probs shape={exist_probs_prior.shape}")

    # --- Debug forward ---
    print("\n=== Debug Forward ===")
    with torch.no_grad():
        model.debug_forward(mif_grid, atom_coords, atom_types, atom_mask, step=5000)

    # --- Save / load test ---
    print("\n=== Save / Load ===")
    model.save("/tmp/test_cvae.pt")
    loaded = ConditionalVAE.load("/tmp/test_cvae.pt")
    print(f"  Loaded model at step {loaded.global_step.item()}")

    # --- Verify config round-trip ---
    assert loaded.cfg.input_shape == cfg.input_shape
    assert loaded.cfg.num_atom_types == cfg.num_atom_types
    assert loaded.cfg.max_atoms == cfg.max_atoms
    assert loaded.cfg.padding_idx == cfg.padding_idx
    assert loaded.cfg.logvar_clamp == cfg.logvar_clamp
    print("  Config round-trip ✓")

    # --- Verify Sinkhorn is used in training, Hungarian in eval ---
    print("\n=== Matching mode ===")
    model.train()
    print(f"  Training mode: use_sinkhorn={model.loss_fn.cfg.use_sinkhorn}, "
          f"self.training={model.training} → will use "
          f"{'Sinkhorn' if model.loss_fn.cfg.use_sinkhorn and model.training else 'Hungarian'}")
    model.eval()
    print(f"  Eval mode:     use_sinkhorn={model.loss_fn.cfg.use_sinkhorn}, "
          f"self.training={model.training} → will use "
          f"{'Sinkhorn' if model.loss_fn.cfg.use_sinkhorn and model.training else 'Hungarian'}")

    print("\n✓ ConditionalVAE test passed!")
