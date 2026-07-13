"""
decoder.py — Conditional decoder for the cVAE.

Takes the ligand latent z and the MIF condition c, and reconstructs
atom coordinates, atom types, and an existence mask.

Architecture:
    [z, c] → 2-layer FC projection → add learned slot embeddings
           → per-atom MLP (with LayerNorm + residuals)
           → coord head (tanh), type head (softmax externally),
             existence head (sigmoid externally)

§-marked parameters MUST be set via DataConfig.to_model_config().
Do not set data-dependent fields manually in DecoderConfig.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

@dataclass
class DecoderConfig:
    """Decoder hyperparameters.

    § Data-dependent fields (default=None) MUST be set via
      DataConfig.to_model_config() → _build_decoder_config().
      Do not set them manually — let DataConfig be the single source of truth.

    Architecture fields have defaults and can be overridden on the
    returned CVaeConfig if needed.
    """

    # --- Data-dependent (REQUIRED from DataConfig) ---
    # § num_atom_types: must match DataConfig.num_atom_types
    #   (includes pad=0 and unk=N+1)
    num_atom_types: Optional[int] = None

    # § max_atoms: must match DataConfig.max_atoms
    max_atoms: Optional[int] = None

    # § latent_dim: must match the ligand encoder's latent_dim
    latent_dim: Optional[int] = None

    # § cond_dim: must match the MIF encoder's output dim (= latent_dim)
    cond_dim: Optional[int] = None

    # § padding_idx: must match DataConfig.pad_id (typically 0)
    padding_idx: Optional[int] = None

    # --- Architecture ---

    # Dimension of the per-atom feature after projecting [z, c]
    hidden_dim: int = 256

    # Learned slot embeddings give each atom position an identity.
    # The decoder learns "slot 0 tends to be carbon, slot 3 tends to be..."
    slot_dim: int = 64

    # Per-atom MLP dimensions (shared across all atom slots).
    # If consecutive dims are equal, residual connections are added automatically.
    per_atom_dims: List[int] = field(default_factory=lambda: [256, 256])

    # --- Coordinate head ---
    # Output coords are in the same normalised space as data.py
    # (typically ≈[-1, 1] with centre_scale normalisation).
    coord_head_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Final activation for coordinates:
    #   nn.Tanh  — for centre_scale coords in [-1, 1]
    #   None     — for unbounded coords (voxel_index or raw Å)
    coord_activation: Optional[str] = "none"

    # --- Atom type head ---
    type_head_dims: List[int] = field(default_factory=lambda: [64])

    # --- Existence head ---
    # Predicts whether each slot is a real atom or padding.
    # Binary logit: sigmoid → probability of existence.
    exist_head_dims: List[int] = field(default_factory=lambda: [64])

    # § Prior probability that a slot contains a real atom.
    #   Used to initialise the existence head bias for balanced training.
    #   Compute from your dataset: mean(real_atom_count / max_atoms).
    #   0.5 is a reasonable default for typical drug-like molecules.
    exist_prior: float = 0.5

    # --- Regularisation ---
    dropout: float = 0.0  # try 0.1–0.2 if overfitting

    # --- FC head init scale ---
    # Small init on output layers so predictions start near zero/centre
    output_init_std: float = 0.01

    def __post_init__(self):
        required = {
            "num_atom_types": self.num_atom_types,
            "max_atoms": self.max_atoms,
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "padding_idx": self.padding_idx,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"DecoderConfig: data-dependent fields {missing} must be set "
                f"via DataConfig.to_model_config(). Do not set them manually."
            )


# ──────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────

class ResMLPBlock(nn.Module):
    """MLP block with LayerNorm, GELU, and optional residual connection.

    If input and output dimensions match, a residual skip connection is added:
        out = GELU(LN(Linear(x))) + x
    Otherwise:
        out = GELU(LN(Linear(x)))
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.has_residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        if self.has_residual:
            h = h + x
        return h


class FCHead(nn.Module):
    """Small FC head with LayerNorm on hidden layers and small output init.

    No activation on the final layer (raw logits or coordinates).
    """

    def __init__(self, dims: List[int], output_init_std: float = 0.01):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # Hidden layers: LN + GELU
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.output_init_std = output_init_std

    def init_output_small(self):
        """Call AFTER main _init_weights to set small init on final layer."""
        last_linear = None
        for m in self.net:
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, 0, self.output_init_std)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
#  Decoder
# ──────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Conditional decoder: [z, c] → ligand (coords, types, existence).

    Architecture:
        1. Concatenate z (ligand latent) and c (MIF condition)
        2. 2-layer FC projection with LayerNorm → hidden representation
        3. Expand to per-atom features + add learned slot embeddings
        4. Shared per-atom MLP with LayerNorm and residual connections
        5. Three heads:
           - Coordinate head (tanh → [-1, 1])
           - Atom type head (raw logits → softmax externally)
           - Existence head (raw logit → sigmoid externally)

    Slot embeddings are critical — they give the decoder a way to
    differentiate atom positions. Without them, every slot would get
    the same input and predict the same atom.

    Note on permutation: the decoder outputs atoms in a fixed slot
    order. The training loss should use Hungarian matching to align
    predicted atoms with target atoms, since PDB atom ordering is
    arbitrary.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.cfg = config

        # --- Condition projection: [z, c] → hidden_dim ---
        # 2-layer MLP with LayerNorm for non-linear z-c interaction
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cfg.latent_dim + self.cfg.cond_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
        )

        # --- Learned slot embeddings ---
        # One vector per atom slot — gives each position a unique identity
        self.slot_embedding = nn.Embedding(self.cfg.max_atoms, self.cfg.slot_dim)

        # --- Per-atom MLP (shared weights, with LayerNorm + residuals) ---
        per_atom_input = self.cfg.hidden_dim + self.cfg.slot_dim
        per_atom_dims = [per_atom_input] + self.cfg.per_atom_dims
        self.per_atom_blocks = nn.ModuleList()
        for i in range(len(per_atom_dims) - 1):
            self.per_atom_blocks.append(ResMLPBlock(
                in_dim=per_atom_dims[i],
                out_dim=per_atom_dims[i + 1],
                dropout=self.cfg.dropout,
            ))

        per_atom_out = self.cfg.per_atom_dims[-1]

        # --- Coordinate head: per_atom_out → 3 ---
        # tanh constrains output to [-1, 1] for centre_scale normalised coords
        coord_dims = [per_atom_out] + self.cfg.coord_head_dims + [3]
        coord_act = self._get_coord_activation()
        self.coord_head = FCHead(coord_dims, output_init_std=self.cfg.output_init_std)
        self.coord_activation = coord_act

        # --- Atom type head: per_atom_out → num_atom_types ---
        type_dims = [per_atom_out] + self.cfg.type_head_dims + [self.cfg.num_atom_types]
        self.type_head = FCHead(type_dims, output_init_std=self.cfg.output_init_std)

        # --- Existence head: per_atom_out → 1 ---
        exist_dims = [per_atom_out] + self.cfg.exist_head_dims + [1]
        self.exist_head = FCHead(exist_dims, output_init_std=self.cfg.output_init_std)

        # --- Weight init ---
        self._init_weights()

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    def _get_coord_activation(self):
        """Resolve coordinate activation from config string."""
        if self.cfg.coord_activation is None or self.cfg.coord_activation == "none":
            return None
        elif self.cfg.coord_activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                f"Unknown coord_activation '{self.cfg.coord_activation}'. "
                f"Use 'tanh' or None."
            )

    def _init_weights(self):
        """Kaiming init for Linear/Conv, small init on output layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small init on output layers so predictions start near zero/centre
        self.coord_head.init_output_small()
        self.type_head.init_output_small()
        self.exist_head.init_output_small()

        # Existence head bias: initialise to log-odds of prior probability
        # so the head starts predicting the dataset average
        # P(exist) = sigmoid(bias) → bias = log(p / (1 - p))
        prior = self.cfg.exist_prior
        if 0.0 < prior < 1.0:
            bias_init = math.log(prior / (1.0 - prior))
            # Find the final Linear in exist_head and set its bias
            for m in reversed(list(self.exist_head.net.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, bias_init)
                    break

    # ──────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        """
        Args:
            z: (B, latent_dim)  — ligand latent code (sampled or from encoder)
            c: (B, cond_dim)    — MIF condition vector (deterministic)

        Returns:
            atom_coords:       (B, max_atoms, 3)              — predicted coordinates
            atom_type_logits:  (B, max_atoms, num_atom_types) — unnormalised logits
            atom_exist_logits: (B, max_atoms, 1)              — existence logit
        """
        B = z.shape[0]
        N = self.cfg.max_atoms

        # 1. Project condition [z, c] → hidden
        cond = torch.cat([z, c], dim=-1)                       # (B, latent+cond)
        cond = self.cond_proj(cond)                             # (B, hidden_dim)

        # 2. Expand to per-atom + add slot embeddings
        cond_expanded = cond.unsqueeze(1).expand(B, N, -1)     # (B, N, hidden_dim)
        slot_ids = torch.arange(N, device=z.device)            # (N,)
        slot_emb = self.slot_embedding(slot_ids)                # (N, slot_dim)
        slot_emb = slot_emb.unsqueeze(0).expand(B, -1, -1)     # (B, N, slot_dim)

        per_atom_input = torch.cat([cond_expanded, slot_emb], dim=-1)  # (B, N, hidden+slot)

        # 3. Shared per-atom MLP (with LayerNorm + residual connections)
        h = per_atom_input
        for block in self.per_atom_blocks:
            h = block(h)                                        # (B, N, per_atom_out)

        # 4. Heads
        atom_coords = self.coord_head(h)                       # (B, N, 3)
        if self.coord_activation is not None:
            atom_coords = self.coord_activation(atom_coords)   # tanh → [-1, 1]

        atom_type_logits = self.type_head(h)                   # (B, N, K)
        atom_exist_logits = self.exist_head(h)                 # (B, N, 1)

        return atom_coords, atom_type_logits, atom_exist_logits

    # ──────────────────────────────────────────
    #  Convenience
    # ──────────────────────────────────────────

    def predict(self, z: torch.Tensor, c: torch.Tensor):
        """
        Convenience method: forward pass + apply activations.

        Returns:
            coords:      (B, max_atoms, 3)              — predicted coordinates
            type_probs:  (B, max_atoms, num_atom_types)  — softmax probabilities
            exist_probs: (B, max_atoms)                  — existence probabilities
        """
        coords, type_logits, exist_logits = self.forward(z, c)
        type_probs = F.softmax(type_logits, dim=-1)
        exist_probs = torch.sigmoid(exist_logits.squeeze(-1))
        return coords, type_probs, exist_probs

    # ──────────────────────────────────────────
    #  Debug
    # ──────────────────────────────────────────

    @torch.no_grad()
    def debug_forward(self, z: torch.Tensor, c: torch.Tensor) -> dict:
        """Forward pass returning structured diagnostics dict."""
        info = {}
        B = z.shape[0]

        # Input
        info["z"] = {
            "shape": list(z.shape),
            "range": [z.min().item(), z.max().item()],
            "mean": z.mean().item(),
            "std": z.std().item(),
        }
        info["c"] = {
            "shape": list(c.shape),
            "range": [c.min().item(), c.max().item()],
            "mean": c.mean().item(),
            "std": c.std().item(),
        }

        # Condition projection
        cond = torch.cat([z, c], dim=-1)
        cond = self.cond_proj(cond)
        info["cond_proj"] = {
            "shape": list(cond.shape),
            "range": [cond.min().item(), cond.max().item()],
            "mean": cond.mean().item(),
            "std": cond.std().item(),
        }

        # Slot embeddings
        slot_ids = torch.arange(self.cfg.max_atoms, device=z.device)
        slot_emb = self.slot_embedding(slot_ids)
        info["slot_emb"] = {
            "shape": list(slot_emb.shape),
            "range": [slot_emb.min().item(), slot_emb.max().item()],
            "mean": slot_emb.mean().item(),
            "std": slot_emb.std().item(),
        }

        # Per-atom MLP
        cond_exp = cond.unsqueeze(1).expand(B, self.cfg.max_atoms, -1)
        slot_exp = slot_emb.unsqueeze(0).expand(B, -1, -1)
        per_atom_in = torch.cat([cond_exp, slot_exp], dim=-1)
        h = per_atom_in
        for i, block in enumerate(self.per_atom_blocks):
            h = block(h)
            info[f"per_atom_block_{i}"] = {
                "shape": list(h.shape),
                "range": [h.min().item(), h.max().item()],
                "mean": h.mean().item(),
                "std": h.std().item(),
            }

        # Heads
        coords, type_logits, exist_logits = self.forward(z, c)

        info["coords"] = {
            "shape": list(coords.shape),
            "range": [coords.min().item(), coords.max().item()],
            "mean": coords.mean().item(),
            "std": coords.std().item(),
        }
        info["type_logits"] = {
            "shape": list(type_logits.shape),
            "range": [type_logits.min().item(), type_logits.max().item()],
            "mean": type_logits.mean().item(),
            "std": type_logits.std().item(),
        }
        info["exist_logits"] = {
            "shape": list(exist_logits.shape),
            "range": [exist_logits.squeeze(-1).min().item(),
                      exist_logits.squeeze(-1).max().item()],
            "mean": exist_logits.mean().item(),
            "std": exist_logits.std().item(),
        }

        # Existence probabilities
        exist_probs = torch.sigmoid(exist_logits.squeeze(-1))
        info["exist_probs"] = {
            "shape": list(exist_probs.shape),
            "mean": exist_probs.mean().item(),
            "std": exist_probs.std().item(),
            "per_sample_mean": exist_probs.mean(dim=1).tolist(),
        }

        # Type probabilities
        type_probs = F.softmax(type_logits, dim=-1)
        info["type_probs"] = {
            "shape": list(type_probs.shape),
            "max_prob_mean": type_probs.max(dim=-1).values.mean().item(),
            "entropy_mean": (
                -(type_probs * (type_probs + 1e-8).log()).sum(dim=-1)
            ).mean().item(),
        }

        # Warnings
        warnings = []
        if torch.isnan(coords).any():
            warnings.append("NaN in predicted coords")
        if torch.isnan(type_logits).any():
            warnings.append("NaN in type logits")
        if torch.isnan(exist_logits).any():
            warnings.append("NaN in exist logits")
        if coords.abs().max() > 1.0 and self.coord_activation is not None:
            warnings.append(f"Coord magnitude > 1 despite tanh (max={coords.abs().max():.4f})")
        info["warnings"] = warnings

        return info

    # ──────────────────────────────────────────
    #  Parameter counts
    # ──────────────────────────────────────────

    def param_counts(self) -> dict:
        """Return per-module parameter counts for profiling."""
        counts = {}
        for name, mod in self.named_children():
            if isinstance(mod, nn.Parameter):
                counts[name] = mod.numel()
            else:
                counts[name] = sum(p.numel() for p in mod.parameters())
        counts["total"] = sum(counts.values())
        return counts


# ──────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Decoder")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--max-atoms", type=int, default=88)
    parser.add_argument("--num-atom-types", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = DecoderConfig(
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
        max_atoms=args.max_atoms,
        num_atom_types=args.num_atom_types,
        padding_idx=0,
    )
    model = Decoder(cfg)
    model.eval()

    # Dummy inputs
    z = torch.randn(args.batch_size, args.latent_dim)
    c = torch.randn(args.batch_size, args.cond_dim)

    # --- Parameter counts ---
    counts = model.param_counts()
    print(f"\n{'='*60}")
    for name, n in counts.items():
        print(f"  {name:25s}  {n:>12,} params")
    print(f"{'='*60}")

    # --- Forward pass ---
    coords, type_logits, exist_logits = model(z, c)
    print(f"\n  Input:    B={args.batch_size}, max_atoms={args.max_atoms}")
    print(f"  coords:   {list(coords.shape)}  range=[{coords.min():.4f}, {coords.max():.4f}]")
    print(f"  types:    {list(type_logits.shape)}  range=[{type_logits.min():.4f}, {type_logits.max():.4f}]")
    print(f"  exist:    {list(exist_logits.shape)}  range=[{exist_logits.squeeze().min():.4f}, "
          f"{exist_logits.squeeze().max():.4f}]")
    print(f"{'='*60}\n")

    # --- Debug info ---
    info = model.debug_forward(z, c)
    for key, val in info.items():
        if isinstance(val, dict):
            parts = [f"shape={val.get('shape', '?')}"]
            if "range" in val:
                parts.append(f"range=[{val['range'][0]:.4f}, {val['range'][1]:.4f}]")
            if "mean" in val:
                parts.append(f"mean={val['mean']:.4f}")
            if "std" in val:
                parts.append(f"std={val['std']:.4f}")
            if "max_prob_mean" in val:
                parts.append(f"max_prob={val['max_prob_mean']:.4f}")
            if "entropy_mean" in val:
                parts.append(f"entropy={val['entropy_mean']:.4f}")
            if "per_sample_mean" in val:
                parts.append(f"per_sample_mean={val['per_sample_mean']}")
            print(f"  {key:25s}  " + "  ".join(parts))
        elif isinstance(val, list) and val:
            print(f"  {key:25s}  ⚠ {'; '.join(val)}")

    # --- Predict convenience ---
    pred_coords, type_probs, exist_probs = model.predict(z, c)
    print(f"\n  predict() output:")
    print(f"    coords      shape={pred_coords.shape}  range=[{pred_coords.min():.4f}, {pred_coords.max():.4f}]")
    print(f"    type_probs  shape={type_probs.shape}  sum(per_slot)={type_probs[0, 0].sum():.4f}")
    print(f"    exist_probs shape={exist_probs.shape}  mean={exist_probs.mean():.4f}")

    # --- Verify coord range ---
    print(f"\n--- Coord activation verification ---")
    print(f"  coord_activation = {cfg.coord_activation}")
    print(f"  coords min={coords.min():.6f}  max={coords.max():.6f}")
    if cfg.coord_activation == "tanh":
        assert coords.min() >= -1.0 and coords.max() <= 1.0, "Coords outside [-1, 1]!"
        print(f"  ✓ Coords within [-1, 1] as expected with tanh")

    # --- Verify existence bias init ---
    print(f"\n--- Existence head bias verification ---")
    prior = cfg.exist_prior
    expected_bias = math.log(prior / (1.0 - prior))
    for m in reversed(list(model.exist_head.net.modules())):
        if isinstance(m, nn.Linear):
            print(f"  exist_prior={prior}  →  expected bias={expected_bias:.4f}  "
                  f"actual bias={m.bias.item():.4f}")
            break

    print("\n✓ Decoder test passed!")
