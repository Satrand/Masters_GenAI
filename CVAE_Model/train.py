#!/usr/bin/env python
"""Training loop for the conditional VAE.

Features:
    - YAML/JSON config file support (with CLI overrides)
    - DataConfig-driven model config (single source of truth)
    - AdamW optimiser with gradient clipping
    - Warmup + cosine LR schedule (per-batch stepping)
    - KL annealing (built into model via step counter)
    - Mixed precision (AMP) with GradScaler
    - Validation with Hungarian matching + reconstruction metrics
    - Latent-space monitoring (active dims, dead dims, scale balance)
    - Step-based + epoch-based checkpointing
    - Optional EMA
    - Console logging + optional TensorBoard
    - Resume from checkpoint
    - NaN loss detection and batch skipping

Usage:
    # From config file:
    python train.py --config experiments/exp01.yaml

    # Config file + CLI overrides:
    python train.py --config experiments/exp01.yaml --lr 3e-4 --epochs 200

    # Generate a default config to edit:
    python train.py --generate-config my_config.yaml

    # Pure CLI:
    python train.py --dataset-dir ./data
"""

import os
import sys
import time
import math
import argparse
import json
from typing import Optional, Dict
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from data_loader import DataConfig, build_dataloaders
from model import ConditionalVAE, CVaeConfig

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def _atom_type_to_int(atom_types):
    """Handle one-hot or int atom types."""
    if atom_types.dim() == 3 and atom_types.size(-1) > 1:
        return atom_types.argmax(dim=-1)
    return atom_types


def make_molecule_figure(coords, types, mask, pred_coords=None, pred_types=None,
                         pred_mask=None, max_mols=4, title_prefix=""):
    """Return a matplotlib Figure of XY-projected molecules.
    
    Args:
        coords:      (B, N, 3) tensor
        types:       (B, N) int or (B, N, T) one-hot
        mask:        (B, N) bool / float
        pred_coords: optional (B, N, 3)
        pred_types:  optional (B, N) int or (B, N, T)
        pred_mask:   optional (B, N) bool / float
    """
    types = _atom_type_to_int(types)
    if pred_types is not None:
        pred_types = _atom_type_to_int(pred_types)

    B = min(coords.size(0), max_mols)
    cols = 2 if pred_coords is not None else 1
    fig, axes = plt.subplots(B, cols, figsize=(4 * cols, 3.5 * B), squeeze=False)
    cmap = plt.cm.get_cmap("tab10")

    for i in range(B):
        # --- Ground truth ---
        ax = axes[i, 0]
        valid = mask[i].bool().cpu()
        c = coords[i][valid].cpu().numpy()
        t = types[i][valid].cpu().numpy()
        for atom_type in sorted(set(t.tolist())):
            idx = t == atom_type
            ax.scatter(c[idx, 0], c[idx, 1], c=[cmap(atom_type % 10)],
                       s=60, alpha=0.8, edgecolors="k", linewidths=0.5,
                       label=f"Type {atom_type}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title_prefix}Input {i}  ({valid.sum().item()} atoms)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")

        # --- Reconstruction / Sample ---
        if pred_coords is not None:
            ax = axes[i, 1]
            if pred_mask is not None:
                p_valid = pred_mask[i].bool().cpu()
            else:
                p_valid = valid
            c_p = pred_coords[i][p_valid].detach().cpu().numpy()
            if pred_types is not None and p_valid.any():
                t_p = pred_types[i][p_valid].detach().cpu().numpy()
                for atom_type in sorted(set(t_p.tolist())):
                    idx = t_p == atom_type
                    ax.scatter(c_p[idx, 0], c_p[idx, 1], c=[cmap(atom_type % 10)],
                               s=60, alpha=0.8, marker="x", linewidths=1.5,
                               label=f"Type {atom_type}")
            else:
                ax.scatter(c_p[:, 0], c_p[:, 1], c="black",
                           s=60, alpha=0.6, marker="x", linewidths=1.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title_prefix}Output {i}  ({p_valid.sum().item()} atoms)")
            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  Latent Monitor (self-contained)
# ──────────────────────────────────────────────

class LatentMonitor:
    """Lightweight latent-space health checks during training.

    Call every validation epoch on a fixed validation batch.
    All outputs are scalars suitable for TensorBoard / console.
    """

    def __init__(self, free_bits_per_dim: float = 0.1,
                 dead_dim_threshold: float = 0.01,
                 logvar_clamp: tuple = (-10.0, 2.0)):
        self.free_bits_per_dim = free_bits_per_dim
        self.dead_dim_threshold = dead_dim_threshold
        self.logvar_clamp = logvar_clamp

    @torch.no_grad()
    def check(self, model, val_batch, device="cuda") -> Dict[str, float]:
        model.eval()
        mif_grid = val_batch["mif_grid"].to(device)
        coords = val_batch["atom_coords"].to(device)
        types = val_batch["atom_types"].to(device)
        mask = val_batch["atom_mask"].to(device)

        c = model.mif_encoder(mif_grid)
        mu, logvar = model.ligand_encoder(coords, types, mask)
        z = model.ligand_encoder.sample_z(coords, types, mask)[0]

        B, latent_dim = mu.shape
        stats = {}

        # KL
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        stats["kl/total"] = kl_per_dim.sum(dim=1).mean().item()
        active = (kl_per_dim > self.free_bits_per_dim).float().sum(dim=1).mean().item()
        stats["kl/active_dims"] = active
        stats["kl/active_pct"] = 100.0 * active / latent_dim

        # Dead dims
        z_std_per_dim = z.std(dim=0)
        n_dead = (z_std_per_dim < self.dead_dim_threshold).sum().item()
        stats["z/dead_dims"] = float(n_dead)
        stats["z/dead_pct"] = 100.0 * n_dead / latent_dim
        stats["z/std"] = z.std().item()
        stats["z/std_per_dim_mean"] = z_std_per_dim.mean().item()
        stats["z/std_per_dim_min"] = z_std_per_dim.min().item()

        # mu / logvar health
        stats["mu/mean"] = mu.mean().item()
        stats["mu/std"] = mu.std().item()
        stats["logvar/mean"] = logvar.mean().item()
        stats["logvar/min"] = logvar.min().item()
        stats["logvar/max"] = logvar.max().item()
        stats["logvar/clamped_low_pct"] = 100.0 * (logvar <= self.logvar_clamp[0]).float().mean().item()
        stats["logvar/clamped_high_pct"] = 100.0 * (logvar >= self.logvar_clamp[1]).float().mean().item()

        # c vs z scale
        c_std = c.std().item()
        z_std = z.std().item()
        stats["scale/c_std"] = c_std
        stats["scale/z_std"] = z_std
        stats["scale/ratio_c_z"] = c_std / (z_std + 1e-8)

        # Decoder existence calibration
        pred_coords, pred_type_logits, pred_exist_logits = model.decoder(z, c)
        exist_probs = torch.sigmoid(pred_exist_logits.squeeze(-1))
        true_occupancy = mask.float().mean().item()
        stats["exist/mean_prob"] = exist_probs.mean().item()
        stats["exist/true_occupancy"] = true_occupancy
        stats["exist/bias_error"] = exist_probs.mean().item() - true_occupancy
        
        stats["_kl_per_dim"] = kl_per_dim.mean(dim=0).cpu().numpy()
        stats["_z_std_per_dim"] = z_std_per_dim.cpu().numpy()
        stats["_mu_mean_per_dim"] = mu.mean(dim=0).cpu().numpy()
        
        model.train()
        
        return stats


# ──────────────────────────────────────────────
#  Simple EMA (self-contained, optional)
# ──────────────────────────────────────────────

class SimpleEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict["decay"]
        self.shadow = {k: v.to(device) for k, v in state_dict["shadow"].items()}


# ──────────────────────────────────────────────
#  Training Config
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    """§ — Training hyperparameters."""

    # --- Data ---
    dataset_dir: str = "./data"
    dataset_path: Optional[str] = None

    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # --- DataConfig overrides (must stay in sync) ---
    num_channels: int = 5
    mif_norm: str = "none"
    coord_norm: str = "none"
    voxel_spacing: float = 0.25
    cache: bool = False
    seed: int = 42

    # --- Model architecture overrides ---
    input_shape: tuple = (128, 128, 128)
    max_atoms: int = 96
    latent_dim: int = 128
    cond_dim: int = 64  
    base_width: int = 16
    hidden_dim: int = 256
    slot_dim: int = 64

    # --- Loss overrides ---
    lambda_coord: float = 1.0
    lambda_type: float = 1.0
    lambda_exist: float = 5.0
    lambda_kl: float = 0.001
    kl_anneal_steps: int = 10000
    kl_schedule: str = "linear"
    free_bits_per_dim: float = 0.1
    coord_loss_type: str = "smooth_l1"

    # --- Optimiser ---
    lr: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    warmup_steps: int = 500
    max_steps: int = 0

    # --- Gradient clipping ---
    max_grad_norm: float = 1.0

    # --- Training ---
    batch_size: int = 16
    epochs: int = 100
    log_every: int = 10
    val_every: int = 1
    save_every: int = 5
    save_every_steps: int = 5000   # step-based checkpoint (0 = disable)
    save_best: bool = True

    # --- Mixed precision ---
    use_amp: bool = True

    # --- EMA ---
    use_ema: bool = False
    ema_decay: float = 0.999

    # --- NaN handling ---
    skip_nan_batches: bool = True

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    resume_from: str = ""

    # --- TensorBoard ---
    use_tensorboard: bool = False
    tb_log_dir: str = "./runs"


# ──────────────────────────────────────────────
#  Config file support (YAML / JSON)
# ──────────────────────────────────────────────

def load_config_file(path: str) -> dict:
    """Load a config file (YAML or JSON) and return as dict."""
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            if not HAS_YAML:
                with open(path, "r") as f2:
                    return json.load(f2)
            f.seek(0)
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0)
                return json.load(f)


def dict_to_trainconfig(cfg_dict: dict) -> TrainConfig:
    """Convert a dict (from YAML/JSON) to a TrainConfig."""
    defaults = asdict(TrainConfig())
    tuple_fields = {"betas", "input_shape"}

    merged = {}
    for field_name, default_val in defaults.items():
        if field_name in cfg_dict:
            val = cfg_dict[field_name]
            if field_name in tuple_fields and isinstance(val, list):
                val = tuple(val)
            merged[field_name] = val
        else:
            merged[field_name] = default_val

    return TrainConfig(**merged)


def save_config_yaml(tcfg: TrainConfig, path: str):
    """Save TrainConfig as a YAML file."""
    if not HAS_YAML:
        json_path = path.replace(".yaml", ".json").replace(".yml", ".json")
        config_dict = {}
        for k, v in asdict(tcfg).items():
            if isinstance(v, tuple):
                v = list(v)
            config_dict[k] = v
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"  (PyYAML not installed — saved as JSON to {json_path})")
        return

    config_dict = {}
    for k, v in asdict(tcfg).items():
        if isinstance(v, tuple):
            v = list(v)
        config_dict[k] = v

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# ──────────────────────────────────────────────
#  Helper: Warmup + Cosine LR schedule
# ──────────────────────────────────────────────

def make_lr_lambda(warmup_steps: int, max_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01
    return lr_lambda


# ──────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────

def _save_checkpoint(path, model, optim, scaler, epoch, best_val_loss, step, ema=None):
    config_dict = {}
    for field_name in model.cfg.__dataclass_fields__:
        val = getattr(model.cfg, field_name)
        if isinstance(val, tuple):
            val = list(val)
        config_dict[field_name] = val

    ckpt = {
        "model_state_dict": model.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "optim_state_dict": optim.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config_dict": config_dict,
        "ema": ema.state_dict() if ema is not None else None,
    }
    torch.save(ckpt, path)


@torch.no_grad()
def _save_latent_snapshot(model, loader, device, checkpoint_dir, epoch, max_batches=50):
    """Save mu and c vectors for offline UMAP analysis."""
    model.eval()
    all_mu, all_c, all_mask = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        mif = batch["mif_grid"].to(device, non_blocking=True)
        coords = batch["atom_coords"].to(device, non_blocking=True)
        types = batch["atom_types"].to(device, non_blocking=True)
        mask = batch["atom_mask"].to(device, non_blocking=True)

        c = model.mif_encoder(mif)
        mu, _ = model.ligand_encoder(coords, types, mask)

        all_mu.append(mu.cpu())
        all_c.append(c.cpu())
        all_mask.append(mask.cpu())

    model.train()

    save_dir = os.path.join(checkpoint_dir, "latents")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "mu": torch.cat(all_mu),
        "c": torch.cat(all_c),
        "mask": torch.cat(all_mask),
        "epoch": epoch,
    }, os.path.join(save_dir, f"epoch_{epoch:04d}.pt"))
    print(f"  Saved latent snapshot: {save_dir}/epoch_{epoch:04d}.pt")


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train_one_epoch(
    model, loader, optim, scaler, scheduler, device, tcfg, epoch, writer=None,
    ema=None, best_val_loss=float("inf")
):
    """Train for one epoch. Returns (avg_losses, best_val_loss)."""
    model.train()
    total_losses = {}
    n_batches = 0
    n_skipped = 0

    for batch_idx, batch in enumerate(loader):
        mif_grid = batch["mif_grid"].to(device, non_blocking=True)
        atom_coords = batch["atom_coords"].to(device, non_blocking=True)
        atom_types = batch["atom_types"].to(device, non_blocking=True)
        atom_mask = batch["atom_mask"].to(device, non_blocking=True)

        step = model.global_step.item()

        with torch.amp.autocast("cuda", enabled=tcfg.use_amp):
            output = model(mif_grid, atom_coords, atom_types, atom_mask, step=step)
            loss = output["total_loss"]
            loss_dict = output["loss_dict"]

        if torch.isnan(loss):
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  ⚠️  NaN loss at step {step}! Skipping batch. "
                      f"(coord={loss_dict.get('coord', '?')}, "
                      f"kl={loss_dict.get('kl', '?')})")
            optim.zero_grad()
            continue

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if tcfg.max_grad_norm > 0:
            scaler.unscale_(optim)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
        else:
            grad_norm = torch.tensor(0.0)

        scaler.step(optim)
        scaler.update()

        if ema is not None:
            ema.update()

        scheduler.step()
        model.increment_step()

        # Accumulate losses (force .item() to avoid GPU leaks)
        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        total_losses["grad_norm"] = total_losses.get("grad_norm", 0.0) + (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
        n_batches += 1

        global_step = model.global_step.item()
        if global_step % tcfg.log_every == 0 or global_step <= 1:
            lr = optim.param_groups[0]["lr"]
            kl_w = loss_dict.get("kl_weight", 0)
            msg = (f"  [epoch {epoch}  step {global_step}]  "
                   f"loss={loss_dict['total']:.4f}  "
                   f"coord={loss_dict['coord']:.4f}  "
                   f"type={loss_dict['type']:.4f}  "
                   f"exist={loss_dict['exist']:.4f}  "
                   f"kl={loss_dict['kl']:.4f}  "
                   f"kl_w={kl_w:.5f}  "
                   f"lr={lr:.2e}  "
                   f"grad={grad_norm:.3f}")

            if "match_coord_dist" in loss_dict:
                msg += f"  m_dist={loss_dict['match_coord_dist']:.4f}"
            if "match_type_acc" in loss_dict:
                msg += f"  m_acc={loss_dict['match_type_acc']:.3f}"

            print(msg)

            if writer is not None:
                writer.add_scalar("train/total_loss", loss_dict["total"], global_step)
                writer.add_scalar("train/coord_loss", loss_dict["coord"], global_step)
                writer.add_scalar("train/type_loss", loss_dict["type"], global_step)
                writer.add_scalar("train/exist_loss", loss_dict["exist"], global_step)
                writer.add_scalar("train/kl_loss", loss_dict["kl"], global_step)
                writer.add_scalar("train/kl_weight", loss_dict.get("kl_weight", 0), global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if "match_coord_dist" in loss_dict:
                    writer.add_scalar("train/match_coord_dist", loss_dict["match_coord_dist"], global_step)
                if "match_type_acc" in loss_dict:
                    writer.add_scalar("train/match_type_acc", loss_dict["match_type_acc"], global_step)
                if "pos_weight" in loss_dict:
                    writer.add_scalar("train/exist_pos_weight", loss_dict["pos_weight"], global_step)
                if "n_real_mean" in loss_dict:
                    writer.add_scalar("train/n_real_mean", loss_dict["n_real_mean"], global_step)

        # Step-based checkpoint
        if tcfg.save_every_steps > 0 and global_step % tcfg.save_every_steps == 0:
            ckpt_path = os.path.join(tcfg.checkpoint_dir, f"step_{global_step:07d}.pt")
            _save_checkpoint(ckpt_path, model, optim, scaler, epoch, best_val_loss, global_step, ema)
            print(f"  Saved step checkpoint: {ckpt_path}")

    if n_batches > 0:
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    else:
        avg_losses = total_losses
    avg_losses["n_skipped"] = n_skipped
    return avg_losses, best_val_loss


@torch.no_grad()
def validate(model, loader, device, step=0):
    """Validate. Returns dict of average losses."""
    model.eval()
    total_losses = {}
    n_batches = 0

    for batch in loader:
        mif_grid = batch["mif_grid"].to(device, non_blocking=True)
        atom_coords = batch["atom_coords"].to(device, non_blocking=True)
        atom_types = batch["atom_types"].to(device, non_blocking=True)
        atom_mask = batch["atom_mask"].to(device, non_blocking=True)

        output = model(mif_grid, atom_coords, atom_types, atom_mask, step=step)
        loss_dict = output["loss_dict"]

        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        n_batches += 1

    if n_batches > 0:
        return {k: v / n_batches for k, v in total_losses.items()}
    return total_losses


# ──────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────

def train(tcfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if tcfg.use_amp and device.type == "cpu":
        print("  Note: AMP enabled but on CPU — disabling.")
        tcfg.use_amp = False

    # Coord-norm warning
    if tcfg.coord_norm == "centre_scale":
        print("\n⚠️  WARNING: coord_norm='centre_scale' centres each ligand independently.\n"
              "    This breaks spatial correspondence with the MIF grid.\n"
              "    Consider coord_norm='none' with raw Å coordinates.\n")

    # Build DataConfig
    data_cfg = DataConfig(
        dataset_dir=tcfg.dataset_dir,
        dataset_path=tcfg.dataset_path,
        num_channels=tcfg.num_channels,
        max_atoms=tcfg.max_atoms,
        mif_norm=tcfg.mif_norm,
        coord_norm=tcfg.coord_norm,
        voxel_spacing=tcfg.voxel_spacing,
        cache=tcfg.cache,
        seed=tcfg.seed,
        expected_grid_shape=tuple(tcfg.input_shape),
        batch_size=tcfg.batch_size,
        num_workers=tcfg.num_workers,
        val_fraction=tcfg.val_split,
        pin_memory=tcfg.pin_memory,
    )

    mcfg = data_cfg.to_model_config()
    mcfg.latent_dim = tcfg.latent_dim #z
    mcfg.cond_dim = tcfg.cond_dim     #c
    mcfg.base_width = tcfg.base_width
    mcfg.hidden_dim = tcfg.hidden_dim
    mcfg.slot_dim = tcfg.slot_dim

    mcfg.lambda_coord = tcfg.lambda_coord
    mcfg.lambda_type = tcfg.lambda_type
    mcfg.lambda_exist = tcfg.lambda_exist
    mcfg.lambda_kl = tcfg.lambda_kl
    mcfg.kl_anneal_steps = tcfg.kl_anneal_steps
    mcfg.kl_schedule = tcfg.kl_schedule
    mcfg.free_bits_per_dim = tcfg.free_bits_per_dim
    mcfg.coord_loss_type = tcfg.coord_loss_type

    model = ConditionalVAE(mcfg).to(device)

    counts = model.count_parameters()
    print(f"\nModel parameters:")
    for name, c in counts.items():
        if name != "total":
            print(f"  {name:15s}  {c['trainable']:>10,}")
    print(f"  {'total':15s}  {counts['total']['trainable']:>10,}")

    train_loader, val_loader = build_dataloaders(data_cfg)

    if train_loader is None:
        print("Training cannot continue without valid data.")
        return

    print(f"\nData:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Batch size:    {tcfg.batch_size}")
    print(f"  num_atom_types: {data_cfg.num_atom_types}")
    print(f"  max_atoms:      {data_cfg.max_atoms}")
    print(f"  input_shape:    {data_cfg.expected_grid_shape}")

    # Optimiser
    optim = AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        betas=tcfg.betas,
    )

    steps_per_epoch = len(train_loader)
    if tcfg.max_steps <= 0:
        tcfg.max_steps = tcfg.epochs * steps_per_epoch
        print(f"  max_steps auto-computed: {tcfg.max_steps}")
    model.loss_fn.annealer.total_steps = tcfg.max_steps
    # EMA
    ema = SimpleEMA(model, decay=tcfg.ema_decay) if tcfg.use_ema else None
    if ema is not None:
        print(f"  EMA enabled (decay={tcfg.ema_decay})")

    # Resume BEFORE scheduler creation
    start_epoch = 1
    best_val_loss = float("inf")
    resumed_step = 0

    if tcfg.resume_from and os.path.isfile(tcfg.resume_from):
        ckpt = torch.load(tcfg.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.global_step.fill_(ckpt["global_step"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        resumed_step = ckpt.get("global_step", 0)

        if "optim_state_dict" in ckpt:
            optim.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "ema" in ckpt and ckpt["ema"] is not None and ema is not None:
            ema.load_state_dict(ckpt["ema"], device=device)

        print(f"\nResumed from {tcfg.resume_from} "
              f"(epoch {start_epoch}, step {resumed_step})")

    # Scheduler created AFTER resume so last_epoch is correct
    lr_lambda = make_lr_lambda(tcfg.warmup_steps, tcfg.max_steps)
    scheduler = LambdaLR(optim, lr_lambda, last_epoch=resumed_step - 1 if resumed_step > 0 else -1)

    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.use_amp)

    # Monitoring
    monitor = LatentMonitor(
        free_bits_per_dim=mcfg.free_bits_per_dim,
        dead_dim_threshold=0.01,
        logvar_clamp=mcfg.logvar_clamp,
    )
    val_monitor_batch = next(iter(val_loader)) if len(val_loader) > 0 else None

    writer = None
    if tcfg.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tcfg.tb_log_dir)
        print(f"TensorBoard logging to {tcfg.tb_log_dir}")

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    effective_config_path = os.path.join(tcfg.checkpoint_dir, "effective_config")
    save_config_yaml(tcfg, effective_config_path + ".yaml")
    config_json = {}
    for k, v in asdict(tcfg).items():
        if isinstance(v, tuple):
            v = list(v)
        config_json[k] = v
    with open(effective_config_path + ".json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Training: {tcfg.epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"  KL anneal: {mcfg.kl_anneal_steps} steps "
          f"(λ_kl: 0 → {mcfg.lambda_kl}, schedule: {mcfg.kl_schedule})")
    print(f"  Free bits: {mcfg.free_bits_per_dim} per dim")
    print(f"  LR: {tcfg.lr}, warmup: {tcfg.warmup_steps} steps")
    print(f"  AMP: {'enabled' if tcfg.use_amp else 'disabled'}")
    print(f"  Matching: {'Sinkhorn (train) + Hungarian (eval)' if mcfg.use_sinkhorn else 'Hungarian'}")
    print(f"  Gradient clip: {tcfg.max_grad_norm}")
    print(f"  NaN skip: {'enabled' if tcfg.skip_nan_batches else 'disabled'}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, tcfg.epochs + 1):
        epoch_start = time.time()

        train_losses, best_val_loss = train_one_epoch(
            model, train_loader, optim, scaler, scheduler, device, tcfg, epoch, writer,
            ema=ema, best_val_loss=best_val_loss
        )

        epoch_time = time.time() - epoch_start
        current_lr = optim.param_groups[0]["lr"]
        current_step = model.global_step.item()
        kl_weight = model.loss_fn.annealer.get_weight(current_step)

        skip_msg = ""
        if train_losses.get("n_skipped", 0) > 0:
            skip_msg = f"  skipped={train_losses['n_skipped']}"
        print(f"\n  Epoch {epoch} train:  "
              f"loss={train_losses['total']:.4f}  "
              f"coord={train_losses['coord']:.4f}  "
              f"type={train_losses['type']:.4f}  "
              f"exist={train_losses['exist']:.4f}  "
              f"kl={train_losses['kl']:.4f}  "
              f"kl_w={kl_weight:.5f}  "
              f"lr={current_lr:.2e}  "
              f"grad={train_losses.get('grad_norm', 0):.3f}  "
              f"time={epoch_time:.1f}s{skip_msg}")

        if "match_coord_dist" in train_losses:
            print(f"                 "
                  f"match_dist={train_losses['match_coord_dist']:.4f}  "
                  f"match_acc={train_losses['match_type_acc']:.3f}  "
                  f"pos_w={train_losses.get('pos_weight', 0):.2f}  "
                  f"n_real={train_losses.get('n_real_mean', 0):.1f}")

        # Validation
        val_losses = {}
        if len(val_loader) > 0 and (epoch % tcfg.val_every == 0 or epoch == tcfg.epochs):
            if ema is not None:
                ema.apply_shadow()

            val_losses = validate(model, val_loader, device, step=current_step)

            print(f"  Epoch {epoch} val:    "
                  f"loss={val_losses['total']:.4f}  "
                  f"coord={val_losses['coord']:.4f}  "
                  f"type={val_losses['type']:.4f}  "
                  f"exist={val_losses['exist']:.4f}  "
                  f"kl={val_losses['kl']:.4f}")

            if "match_coord_dist" in val_losses:
                print(f"                 "
                      f"match_dist={val_losses['match_coord_dist']:.4f}  "
                      f"match_acc={val_losses['match_type_acc']:.3f}")

            # ── Latent monitor ──
            if val_monitor_batch is not None:
                monitor_stats = monitor.check(model, val_monitor_batch, device=device)
                print(f"  Latent: active={monitor_stats['kl/active_pct']:.1f}% | "
                      f"dead={monitor_stats['z/dead_dims']:.0f}/{mcfg.latent_dim} | "
                      f"z_std={monitor_stats['z/std']:.3f} | "
                      f"c/z_ratio={monitor_stats['scale/ratio_c_z']:.2f}")

                if monitor_stats["kl/active_pct"] < 15.0 and current_step > mcfg.kl_anneal_steps:
                    print(f"  ⚠️  POSTERIOR COLLAPSE: only {monitor_stats['kl/active_pct']:.1f}% active!")
                if monitor_stats["scale/ratio_c_z"] > 5.0 or monitor_stats["scale/ratio_c_z"] < 0.2:
                    print(f"  ⚠️  SCALE MISMATCH: c/z ratio = {monitor_stats['scale/ratio_c_z']:.2f}")

                if writer is not None:
                    for k, v in monitor_stats.items():
                        if not k.startswith("_"):
                            writer.add_scalar(f"latent/{k}", v, current_step)

                    # ── NEW: histograms ──
                    writer.add_histogram("latent/kl_per_dim",
                                         monitor_stats.get("_kl_per_dim",
                                                           torch.zeros(mcfg.latent_dim)),
                                         current_step)
                    writer.add_histogram("latent/z_std_per_dim",
                                         monitor_stats.get("_z_std_per_dim",
                                                           torch.zeros(mcfg.latent_dim)),
                                         current_step)
                    writer.add_histogram("latent/mu_mean_per_dim",
                                         monitor_stats.get("_mu_mean_per_dim",
                                                           torch.zeros(mcfg.latent_dim)),
                                         current_step)

            # ── NEW: molecular visualisations ──
            if writer is not None and epoch % max(1, tcfg.val_every * 5) == 0:
                model.eval()
                with torch.no_grad():
                    # Grab one validation batch
                    viz_batch = val_monitor_batch
                    mif = viz_batch["mif_grid"].to(device)
                    coords = viz_batch["atom_coords"].to(device)
                    types = viz_batch["atom_types"].to(device)
                    mask = viz_batch["atom_mask"].to(device)

                    # Reconstructions
                    out = model(mif, coords, types, mask, step=current_step)
                    pred_coords = out["pred_coords"]
                    pred_types = out.get("pred_types")
                    pred_exist = torch.sigmoid(out.get("pred_exist_logits", torch.zeros_like(mask)).squeeze(-1))

                    # Threshold existence at 0.5 for plotting
                    pred_mask = pred_exist > 0.5

                    fig = make_molecule_figure(
                        coords, types, mask,
                        pred_coords=pred_coords,
                        pred_types=pred_types,
                        pred_mask=pred_mask,
                        max_mols=4,
                        title_prefix="Recon "
                    )
                    writer.add_figure("val/reconstructions_xy", fig, current_step)
                    plt.close(fig)

                    # Prior samples (decode from N(0,I))
                    z_sample = torch.randn(4, model.cfg.latent_dim, device=device)
                    c_sample = model.mif_encoder(mif[:4])
                    p_coords, p_types_logits, p_exist_logits = model.decoder(z_sample, c_sample)
                    p_types = p_types_logits.argmax(dim=-1) if p_types_logits.dim() == 3 else None
                    p_mask = torch.sigmoid(p_exist_logits.squeeze(-1)) > 0.5

                    fig = make_molecule_figure(
                        p_coords,
                        p_types if p_types is not None else torch.zeros_like(p_mask).long(),
                        p_mask,
                        max_mols=4,
                        title_prefix="Prior "
                    )
                    writer.add_figure("val/prior_samples_xy", fig, current_step)
                    plt.close(fig)

                model.train()

            if writer is not None:
                for k, v in val_losses.items():
                    writer.add_scalar(f"val/{k}", v, current_step)

            if tcfg.save_best and val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                best_path = os.path.join(tcfg.checkpoint_dir, "best_model.pt")
                _save_checkpoint(best_path, model, optim, scaler, epoch, best_val_loss, current_step, ema)
                print(f"  ★ New best model! val_loss={best_val_loss:.4f}")

            if epoch % 10 == 0:
                _save_latent_snapshot(model, val_loader, device, tcfg.checkpoint_dir, epoch)

            if ema is not None:
                ema.restore()

        # Epoch checkpoint
        if epoch % tcfg.save_every == 0 or epoch == tcfg.epochs:
            ckpt_path = os.path.join(tcfg.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            _save_checkpoint(ckpt_path, model, optim, scaler, epoch, best_val_loss, current_step, ema)
            print(f"  Saved checkpoint: {ckpt_path}")

        if writer is not None:
            writer.add_scalar("epoch/train_total_loss", train_losses["total"], epoch)
            writer.add_scalar("epoch/val_total_loss", val_losses.get("total", 0), epoch)
            writer.add_scalar("epoch/kl_weight", kl_weight, epoch)
            writer.add_scalar("epoch/lr", current_lr, epoch)
            writer.flush()

    if writer is not None:
        writer.close()

    print(f"\n{'='*70}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {tcfg.checkpoint_dir}")
    print(f"  Total steps: {model.global_step.item()}")
    print(f"{'='*70}")

    return model


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train conditional VAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config experiments/exp01.yaml
  python train.py --config experiments/exp01.yaml --lr 3e-4 --epochs 200
  python train.py --generate-config my_config.yaml
  python train.py --dataset-dir ./data
        """)

    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML or JSON config file.")
    parser.add_argument("--generate-config", type=str, default="",
                        help="Generate a default config file and exit.")

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=None)

    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--mif-norm", type=str, default=None)
    parser.add_argument("--coord-norm", type=str, default=None)
    parser.add_argument("--voxel-spacing", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--input-shape", type=int, nargs=3, default=None)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--cond-dim", type=int, default=None)
    parser.add_argument("--base-width", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--slot-dim", type=int, default=None)

    parser.add_argument("--lambda-coord", type=float, default=None)
    parser.add_argument("--lambda-type", type=float, default=None)
    parser.add_argument("--lambda-exist", type=float, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--kl-anneal-steps", type=int, default=None)
    parser.add_argument("--kl-schedule", type=str, default=None,
                        choices=["linear", "sigmoid", "cyclical"])
    parser.add_argument("--free-bits", type=float, default=None)
    parser.add_argument("--coord-loss-type", type=str, default=None,
                        choices=["smooth_l1", "mse"])

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--use-ema", action="store_true")

    args = parser.parse_args()

    if args.generate_config:
        default_cfg = TrainConfig()
        save_config_yaml(default_cfg, args.generate_config)
        print(f"Default config saved to {args.generate_config}")
        sys.exit(0)

    if args.config:
        print(f"Loading config from {args.config}")
        cfg_dict = load_config_file(args.config)
        tcfg = dict_to_trainconfig(cfg_dict)
        print(f"  Loaded {len(cfg_dict)} fields from config file")
    else:
        tcfg = TrainConfig()

    overrides = {
        "dataset_dir": args.dataset_dir,
        "dataset_path": args.dataset_path,
        "val_split": args.val_split,
        "num_channels": args.num_channels,
        "mif_norm": args.mif_norm,
        "coord_norm": args.coord_norm,
        "voxel_spacing": args.voxel_spacing,
        "seed": args.seed,
        "input_shape": tuple(args.input_shape) if args.input_shape else None,
        "max_atoms": args.max_atoms,
        "latent_dim": args.latent_dim,
        "cond_dim": args.cond_dim,
        "base_width": args.base_width,
        "hidden_dim": args.hidden_dim,
        "slot_dim": args.slot_dim,
        "lambda_coord": args.lambda_coord,
        "lambda_type": args.lambda_type,
        "lambda_exist": args.lambda_exist,
        "lambda_kl": args.lambda_kl,
        "kl_anneal_steps": args.kl_anneal_steps,
        "kl_schedule": args.kl_schedule,
        "free_bits_per_dim": args.free_bits,
        "coord_loss_type": args.coord_loss_type,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "log_every": args.log_every,
        "val_every": args.val_every,
        "save_every": args.save_every,
        "num_workers": args.num_workers,
        "checkpoint_dir": args.checkpoint_dir,
        "resume_from": args.resume_from,
    }

    n_overrides = 0
    for k, v in overrides.items():
        if v is not None:
            setattr(tcfg, k, v)
            n_overrides += 1

    if args.no_amp:
        tcfg.use_amp = False
    if args.use_tensorboard:
        tcfg.use_tensorboard = True
    if args.use_ema:
        tcfg.use_ema = True
    if args.no_cache:
        tcfg.cache = False

    if args.config and n_overrides > 0:
        print(f"  Applied {n_overrides} CLI overrides")

    if not tcfg.dataset_dir:
        parser.error("--dataset-dir is required (via CLI or config file)")

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    effective_yaml = os.path.join(tcfg.checkpoint_dir, "effective_config.yaml")
    save_config_yaml(tcfg, effective_yaml)
    config_json = {}
    for k, v in asdict(tcfg).items():
        if isinstance(v, tuple):
            v = list(v)
        config_json[k] = v
    with open(os.path.join(tcfg.checkpoint_dir, "effective_config.json"), "w") as f:
        json.dump(config_json, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"  Effective training config")
    print(f"{'='*50}")
    for k, v in asdict(tcfg).items():
        print(f"  {k:25s} = {v}")
    print(f"{'='*50}\n")

    train(tcfg)
