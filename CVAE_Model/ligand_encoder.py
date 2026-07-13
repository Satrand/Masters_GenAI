"""
ligand_encoder.py — PointNet/EdgeConv ligand encoder for conditional VAE.

Takes a set of atoms (coords + type) and encodes them into a latent
distribution (mu, logvar) suitable for the VAE reparameterisation trick.

Supports two architectures:
  - EdgeConv (default): k-NN graph convolution for inter-atomic reasoning.
    First layer uses coordinate-space k-NN (physical neighbourhoods);
    subsequent layers optionally use feature-space k-NN (DGCNN dynamic graph).
  - PointNet (legacy): shared MLP per atom, no inter-atomic communication.
    Kept for ablation studies.

Architecture (EdgeConv mode):
    atom_coords (B, N, 3)  ─┐
                             ├─→ Embed+Concat → EdgeConv₁ → EdgeConv₂ → ... → Pool → Norm → FC → μ, logvar
    atom_types  (B, N)    ─┘

    EdgeConv block:
        For each atom i:
          1. Find k nearest neighbours (coordinate- or feature-space)
          2. Compute edge features: [x_i ∥ (x_j − x_i)]  for each neighbour j
          3. Apply shared MLP to edge features
          4. Aggregate over neighbours (max or mean)
          5. Add residual connection

Key design choices:
    - EdgeConv captures local inter-atomic structure (distances, angles)
      that a plain PointNet MLP cannot learn
    - DGCNN dynamic graph: layer 1 uses physical coordinates for k-NN,
      later layers use learned features (captures higher-order structure)
    - Masking: padding atoms excluded from k-NN search, neighbour
      aggregation, and output zeroed after each block
    - Separate LayerNorm for max-pool and avg-pool (preserves distinct
      statistics of each pooling mode)
    - Small init on FC head outputs so μ, logvar ≈ 0 at start (z ≈ N(0,I))
    - logvar clamping prevents posterior collapse / explosion
    - learnable emb_scale balances coordinate vs. embedding magnitude
    - sample_from_prior() for generation; deterministic flag decoupled
      from self.training

§-marked parameters should be updated to match your dataset.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
#  Shared VAE utilities
# ──────────────────────────────────────────────

def reparameterise(mu: torch.Tensor, logvar: torch.Tensor,
                   deterministic: bool = False) -> torch.Tensor:
    """Reparameterisation trick: z = mu + std * eps, eps ~ N(0,I).

    Args:
        mu:           Mean of latent distribution (B, latent_dim)
        logvar:       Log-variance of latent distribution (B, latent_dim)
        deterministic: If True, return mu directly (no noise)

    Returns:
        z: Sampled latent vector (B, latent_dim)
    """
    if deterministic:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

@dataclass
class LigandEncoderConfig:
    """§ — Update these to match your dataset and model design."""

    # --- Atom vocabulary ---
    num_atom_types: int = 13
    padding_idx: int = 0

    # --- Embedding ---
    atom_type_emb_dim: int = 32       # § 32 is fine for ≤15 types

    # --- PointNet body (legacy, use_edge_conv=False) ---
    point_dims: list = field(default_factory=lambda: [128, 256])

    # --- EdgeConv (use_edge_conv=True) ---
    use_edge_conv: bool = True
    edge_conv_k: int = 8              # § neighbours for k-NN graph
    edge_conv_dims: list = field(default_factory=lambda: [128, 256])
    edge_conv_aggr: str = "max"       # "max" or "mean"
    dynamic_graph: bool = True        # § True = feature-space k-NN after layer 1 (DGCNN)

    # --- Pooling ---
    pool_mode: str = "maxavg"         # "maxavg" (concat max+avg) or "max"

    # --- FC head (after pool) ---
    head_dims: list = field(default_factory=lambda: [256])

    # --- Latent ---
    latent_dim: int = 128             # § must match MIF encoder

    # --- Regularisation ---
    dropout: float = 0.0              # § try 0.1–0.2 if overfitting
    logvar_clamp: tuple = (-4.0, 2.0)  # § clamp logvar to prevent collapse/explosion

    # --- Data ---
    max_atoms: int = 96               # § must match data.py

    # --- FC head init scale ---
    fc_init_std: float = 0.01


# ──────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────

class MaskedMLP(nn.Module):
    """MLP with LayerNorm and optional masking for padding atoms.

    Kept for legacy PointNet mode (use_edge_conv=False).
    """

    def __init__(self, dims, dropout=0.0, use_norm=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, N, D_in)
            mask: (B, N)  True = real atom, False = padding
        Returns:
            (B, N, D_out)
        """
        out = self.net(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


class EdgeConvBlock(nn.Module):
    """EdgeConv layer: k-NN graph → edge features → shared MLP → aggregate.

    For each atom i, finds k nearest neighbours, computes edge features
    [x_i || (x_j - x_i)], applies a shared MLP, and aggregates with
    max or mean pooling over neighbours. Includes a residual connection.

    The k-NN graph can be computed in coordinate space (physical neighbours)
    or feature space (learned neighbours — DGCNN dynamic graph).

    Args:
        in_dim:   Input per-atom feature dimension
        out_dim:  Output per-atom feature dimension
        k:        Number of nearest neighbours
        aggr:     Aggregation over neighbours ('max' or 'mean')
        dropout:  Dropout rate
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 8,
                 aggr: str = "max", dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.aggr = aggr

        # Edge MLP: [x_i || (x_j - x_i)] = 2*in_dim → out_dim
        # Two layers for expressivity; LayerNorm + GELU throughout.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Residual connection (project if dims change)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for edge MLP; small init for residual projection."""
        for m in self.edge_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Small init on residual projection so output is dominated by
        # the edge MLP at init (the learned path), with the residual
        # growing over training to improve gradient flow.
        if isinstance(self.residual, nn.Linear):
            nn.init.normal_(self.residual.weight, 0, 0.01)
            nn.init.zeros_(self.residual.bias)

    # --------------------------------------------------------------- k-NN

    def _compute_knn(self, query: torch.Tensor, reference: torch.Tensor,
                     atom_mask: torch.Tensor) -> torch.Tensor:
        """Compute k nearest neighbours using pairwise Euclidean distances.

        Padding atoms (atom_mask=False) are excluded from the neighbour
        search by setting their distances to inf. Self-loops are also
        excluded.

        Args:
            query:     (B, N, D) features for distance computation
            reference: (B, N, D) reference features (same as query for self-k-NN)
            atom_mask: (B, N) True = real atom

        Returns:
            knn_idx: (B, N, k) indices of k nearest real neighbours (no self)
        """
        B, N, _ = query.shape

        # Full pairwise distance matrix — O(N²) but N≤64, so negligible
        dist = torch.cdist(query, reference)  # (B, N, N)

        # Mask padding atoms: set distance to inf so they're never selected
        pad_mask = ~atom_mask  # (B, N) True = padding
        dist = dist.masked_fill(pad_mask.unsqueeze(1), float('inf'))  # row: query is padding
        dist = dist.masked_fill(pad_mask.unsqueeze(2), float('inf'))  # col: ref is padding

        # Exclude self: set diagonal to inf
        diag = torch.arange(N, device=query.device)
        dist[:, diag, diag] = float('inf')

        # k nearest neighbours (smallest distances)
        knn_idx = dist.topk(self.k, dim=-1, largest=False).indices  # (B, N, k)
        return knn_idx

    # ----------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                atom_mask: torch.Tensor, graph_coords: bool = True) -> torch.Tensor:
        """
        Args:
            x:           (B, N, D_in)  per-atom features
            coords:      (B, N, 3)     atom coordinates (for coord-space k-NN)
            atom_mask:   (B, N)        True = real atom
            graph_coords: If True, k-NN in coordinate space (physical neighbours).
                          If False, k-NN in feature space (learned neighbours, DGCNN).

        Returns:
            (B, N, D_out) updated per-atom features
        """
        B, N, D = x.shape
        k = self.k

        # --- k-NN graph ---
        if graph_coords:
            knn_idx = self._compute_knn(coords, coords, atom_mask)
        else:
            knn_idx = self._compute_knn(x, x, atom_mask)

        # --- Gather neighbour features (fancy indexing) ---
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        x_neighbors = x[batch_idx, knn_idx]  # (B, N, k, D)

        # --- Edge features: [center || (neighbour − center)] ---
        x_center = x.unsqueeze(2).expand_as(x_neighbors)  # (B, N, k, D)
        edge_feat = torch.cat([x_center, x_neighbors - x_center], dim=-1)  # (B, N, k, 2D)

        # --- Shared edge MLP ---
        edge_out = self.edge_mlp(edge_feat)  # (B, N, k, out_dim)

        # --- Aggregate over neighbours (with padding masking) ---
        # Some k-NN slots may point to padding atoms (when #real < k+1).
        # Check which neighbours are real atoms.
        #neighbor_is_real = atom_mask.gather(1, knn_idx)  # (B, N, k)
        neighbor_is_real = atom_mask.unsqueeze(-1).expand(-1, -1, knn_idx.size(-1)).gather(1, knn_idx)

        if self.aggr == 'max':
            # Replace padding neighbours with -inf so they never win the max
            edge_out = edge_out.masked_fill(~neighbor_is_real.unsqueeze(-1), float('-inf'))
            x_out = edge_out.max(dim=2).values  # (B, N, out_dim)
            # Safety: if ALL neighbours were padding (shouldn't happen), replace NaN
            x_out = torch.nan_to_num(x_out, nan=0.0, posinf=0.0, neginf=0.0)
        else:  # mean
            neighbor_mask = neighbor_is_real.unsqueeze(-1).float()  # (B, N, k, 1)
            x_out = (edge_out * neighbor_mask).sum(dim=2) / neighbor_mask.sum(dim=2).clamp(min=1)

        # --- Residual connection ---
        x_out = x_out + self.residual(x)

        # --- Zero out padding atoms ---
        x_out = x_out * atom_mask.unsqueeze(-1).float()

        return x_out


class FCHead(nn.Module):
    """FC head with optional hidden layers and small output init."""

    def __init__(self, dims, output_init_std=0.01):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on final layer
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self._output_init_std = output_init_std

    def init_output_small(self):
        """Small init on final layer so mu/logvar start near zero.
        Call AFTER the main _init_weights to override it."""
        last_linear = None
        for m in self.net:
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, 0, self._output_init_std)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        return self.net(x)
# ──────────────────────────────────────────────
#  Ligand Encoder
# ──────────────────────────────────────────────

class LigandEncoder(nn.Module):
    """PointNet/EdgeConv encoder for 3D ligand structures.

    EdgeConv mode (default):
        atom_coords (B,N,3)  ─┐
                                 ├─→ Embed+Concat → EdgeConv×L → Max+AvgPool → LayerNorm → FC → μ, logvar
        atom_types  (B,N)    ─┘

    PointNet mode (legacy, for ablation):
        atom_coords (B,N,3)  ─┐
                                 ├─→ Embed+Concat → MaskedMLP → Max+AvgPool → LayerNorm → FC → μ, logvar
        atom_types  (B,N)    ─┘

    Key properties:
        - Permutation invariant (same molecule → same z regardless of atom order)
        - Mask-aware (padding atoms never contribute to k-NN or aggregation)
        - Max+Avg pooling with SEPARATE LayerNorm (preserves distinct pooling stats)
        - EdgeConv: k-NN graph captures local inter-atomic structure
        - DGCNN: dynamic graph (feature-space k-NN) after first layer
        - Debug-friendly (debug_forward returns structured dict)
    """

    def __init__(self, config: LigandEncoderConfig = None):
        super().__init__()
        self.cfg = config or LigandEncoderConfig()

        if self.cfg.use_edge_conv and len(self.cfg.edge_conv_dims) == 0:
            raise ValueError("edge_conv_dims must have at least one element when use_edge_conv=True")

        # --- Atom type embedding ---
        self.atom_embedding = nn.Embedding(
            self.cfg.num_atom_types,
            self.cfg.atom_type_emb_dim,
            padding_idx=self.cfg.padding_idx,
        )
        # Learnable scale so embeddings match coordinate magnitude
        self.emb_scale = nn.Parameter(torch.tensor(1.0))

        # --- Per-atom feature processing ---
        feat_dim = 3 + self.cfg.atom_type_emb_dim  # coords + type embedding

        if self.cfg.use_edge_conv:
            # EdgeConv blocks: input_dims[i] → edge_conv_dims[i]
            ec_in_dims = [feat_dim] + self.cfg.edge_conv_dims[:-1]
            ec_out_dims = self.cfg.edge_conv_dims

            self.edge_conv_blocks = nn.ModuleList([
                EdgeConvBlock(
                    in_dim=in_d, out_dim=out_d,
                    k=self.cfg.edge_conv_k,
                    aggr=self.cfg.edge_conv_aggr,
                    dropout=self.cfg.dropout,
                )
                for in_d, out_d in zip(ec_in_dims, ec_out_dims)
            ])
            self.point_mlp = None  # not used
            per_atom_out_dim = self.cfg.edge_conv_dims[-1]
        else:
            # Legacy PointNet: shared MaskedMLP
            point_dims = [feat_dim] + self.cfg.point_dims
            self.point_mlp = MaskedMLP(point_dims, dropout=self.cfg.dropout, use_norm=True)
            self.edge_conv_blocks = None  # not used
            per_atom_out_dim = self.cfg.point_dims[-1]

        # --- Pooling with SEPARATE normalisation for max and avg ---
        # Max and avg pooling produce features on different scales;
        # normalising them independently preserves their distinct statistics.
        self.max_norm = nn.LayerNorm(per_atom_out_dim)
        if self.cfg.pool_mode == "maxavg":
            self.avg_norm = nn.LayerNorm(per_atom_out_dim)
            pooled_dim = per_atom_out_dim * 2
        else:
            self.avg_norm = None
            pooled_dim = per_atom_out_dim

        # --- FC heads: pooled feature → mu, logvar ---
        head_dims = [pooled_dim] + self.cfg.head_dims + [self.cfg.latent_dim]
        self.head_mu = FCHead(head_dims, output_init_std=self.cfg.fc_init_std)
        self.head_logvar = FCHead(head_dims, output_init_std=self.cfg.fc_init_std)

        self._init_weights()

    # ---------------------------------------------------------------- init

    def _init_weights(self):
        """Initialise weights: Kaiming for Linears, normal for Embeddings,
        small init on FC head outputs (μ, logvar ≈ 0 at start)."""
    
        # 1. General init — skip EdgeConvBlock subgraphs so we don't
        #    clobber their custom residual init.
        for m in self.modules():
            if isinstance(m, EdgeConvBlock):
                continue
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 2. Re-apply EdgeConvBlock init (residual small-init)
        if self.edge_conv_blocks is not None:
            for block in self.edge_conv_blocks:
                block._init_weights()

        # 3. Zero out padding embedding (safer outside the loop)
        with torch.no_grad():
            self.atom_embedding.weight[self.cfg.padding_idx].fill_(0)

        # 4. Override FC head outputs: small init so μ, logvar ≈ 0 at start
        self.head_mu.init_output_small()
        self.head_logvar.init_output_small()
        

    def _pool(self, point_feat: torch.Tensor,
              atom_mask: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant pooling over atoms with separate normalisation.

        Max-pool and avg-pool are normalised independently before
        concatenation, preserving their distinct statistical properties.

        Args:
            point_feat: (B, N, D) per-atom features
            atom_mask:  (B, N) True = real atom

        Returns:
            (B, D') global feature — D' = D if max, 2D if maxavg
        """
        mask_float = atom_mask.unsqueeze(-1).float()  # (B, N, 1)

        # Max pool — replace padding with -inf
        global_max = torch.where(
            atom_mask.unsqueeze(-1),
            point_feat,
            torch.tensor(float('-inf'), device=point_feat.device, dtype=point_feat.dtype),
        ).max(dim=1).values  # (B, D)
        global_max = torch.nan_to_num(global_max, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise max separately
        global_max = self.max_norm(global_max)

        if self.cfg.pool_mode == "max":
            return global_max

        # Avg pool — sum real atoms, divide by count
        global_avg = (point_feat * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        # Normalise avg separately
        global_avg = self.avg_norm(global_avg)

        return torch.cat([global_max, global_avg], dim=-1)  # (B, 2D)

    # ----------------------------------------------------------- forward

    def forward(self, atom_coords: torch.Tensor, atom_types: torch.Tensor,
                atom_mask: torch.Tensor):
        """
        Args:
            atom_coords: (B, N, 3) float32 — normalised coordinates
            atom_types:  (B, N)    long    — atom type indices
            atom_mask:   (B, N)    bool    — True = real atom

        Returns:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Embed atom types (scaled to match coordinate magnitude)
        type_emb = self.atom_embedding(atom_types) * self.emb_scale  # (B, N, emb_dim)

        # Concatenate coords + type embedding
        point_feat = torch.cat([atom_coords, type_emb], dim=-1)  # (B, N, 3+emb)

        # Per-atom feature processing
        if self.cfg.use_edge_conv:
            for i, block in enumerate(self.edge_conv_blocks):
                # Layer 0: coordinate-space k-NN (physical neighbourhoods)
                # Layer 1+: feature-space k-NN (DGCNN dynamic graph)
                use_coords = (i == 0) or (not self.cfg.dynamic_graph)
                point_feat = block(point_feat, atom_coords, atom_mask,
                                   graph_coords=use_coords)
        else:
            point_feat = self.point_mlp(point_feat, mask=atom_mask)

        # Permutation-invariant pooling
        global_feat = self._pool(point_feat, atom_mask)

        # FC heads → distribution parameters
        mu = self.head_mu(global_feat)
        logvar = self.head_logvar(global_feat)
        logvar = logvar.clamp(*self.cfg.logvar_clamp)

        return mu, logvar

    # ----------------------------------------------------------- sampling

    def sample_z(self, atom_coords, atom_types, atom_mask, deterministic=None):
        """Encode + reparameterise.

        Args:
            atom_coords, atom_types, atom_mask: same as forward()
            deterministic: If True, return mu (no noise). If False, sample.
                           If None (default), uses not self.training.

        Returns:
            z:      (B, latent_dim) sampled latent
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.forward(atom_coords, atom_types, atom_mask)
        if deterministic is None:
            deterministic = not self.training
        z = reparameterise(mu, logvar, deterministic=deterministic)
        return z, mu, logvar

    def sample_from_prior(self, batch_size: int, device=None):
        """Sample z ~ N(0, I) — for unconditional or conditional generation
        (combine with c from the MIF encoder for conditional generation)."""
        return torch.randn(batch_size, self.cfg.latent_dim, device=device)

    # ------------------------------------------------------ param counts

    def param_counts(self) -> dict:
        """Return per-module parameter counts for profiling.
        Fixes bug where top-level Parameters (e.g. emb_scale) were missed."""
        counts = {}
        for name, mod in self.named_children():
            counts[name] = sum(p.numel() for p in mod.parameters())
        # Catch standalone Parameters (e.g. emb_scale)
        for name, param in self.named_parameters():
            if '.' not in name:  # top-level parameter
                counts[name] = param.numel()
        counts['total'] = sum(counts.values())
        return counts

    # ------------------------------------------------------------- debug

    @torch.no_grad()
    def debug_forward(self, atom_coords, atom_types, atom_mask):
        """Forward pass returning structured diagnostics dict."""
        info = {}
        B, N, _ = atom_coords.shape

        real_coords = atom_coords[atom_mask]
        info["input"] = {
            "shape": [B, N],
            "range": [real_coords.min().item(), real_coords.max().item()],
            "real_atoms": atom_mask.sum(1).tolist(),
            "unique_types": atom_types[atom_mask].unique().tolist(),
        }

        # Embed
        type_emb = self.atom_embedding(atom_types) * self.emb_scale
        real_emb = type_emb[atom_mask]
        info["type_emb"] = {
            "shape": list(type_emb.shape),
            "range": [real_emb.min().item(), real_emb.max().item()],
            "mean": real_emb.mean().item(),
            "std": real_emb.std().item(),
        }

        # Concat
        point_feat = torch.cat([atom_coords, type_emb], dim=-1)
        info["point_input"] = {"shape": list(point_feat.shape)}

        # Per-atom processing
        if self.cfg.use_edge_conv:
            for i, block in enumerate(self.edge_conv_blocks):
                use_coords = (i == 0) or (not self.cfg.dynamic_graph)
                point_feat = block(point_feat, atom_coords, atom_mask,
                                   graph_coords=use_coords)
                real_feats = point_feat[atom_mask]
                graph_type = "coords" if use_coords else "features"
                info[f"edge_conv_{i}"] = {
                    "shape": list(point_feat.shape),
                    "range": [real_feats.min().item(), real_feats.max().item()],
                    "mean": real_feats.mean().item(),
                    "std": real_feats.std().item(),
                    "graph": graph_type,
                    "k": block.k,
                }
        else:
            point_feat = self.point_mlp(point_feat, mask=atom_mask)
            real_feats = point_feat[atom_mask]
            info["point_mlp"] = {
                "shape": list(point_feat.shape),
                "range": [real_feats.min().item(), real_feats.max().item()],
                "mean": real_feats.mean().item(),
                "std": real_feats.std().item(),
            }

        # Pool (separate max/avg for diagnostics)
        mask_float = atom_mask.unsqueeze(-1).float()
        global_max = torch.where(
            atom_mask.unsqueeze(-1),
            point_feat,
            torch.tensor(float('-inf'), device=point_feat.device, dtype=point_feat.dtype),
        ).max(dim=1).values
        global_max = torch.nan_to_num(global_max, nan=0.0, posinf=0.0, neginf=0.0)
        global_avg = (point_feat * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        info["pool_max"] = {
            "shape": list(global_max.shape),
            "range": [global_max.min().item(), global_max.max().item()],
            "mean": global_max.mean().item(),
            "std": global_max.std().item(),
        }
        info["pool_avg"] = {
            "shape": list(global_avg.shape),
            "range": [global_avg.min().item(), global_avg.max().item()],
            "mean": global_avg.mean().item(),
            "std": global_avg.std().item(),
        }

        # Pool norm
        global_feat = self._pool(point_feat, atom_mask)
        info["pool_norm"] = {
            "shape": list(global_feat.shape),
            "range": [global_feat.min().item(), global_feat.max().item()],
            "mean": global_feat.mean().item(),
            "std": global_feat.std().item(),
        }

        # Heads
        mu = self.head_mu(global_feat)
        logvar = self.head_logvar(global_feat)
        logvar_raw = logvar.clone()
        logvar = logvar.clamp(*self.cfg.logvar_clamp)

        info["mu"] = {
            "shape": list(mu.shape),
            "range": [mu.min().item(), mu.max().item()],
            "mean": mu.mean().item(),
            "std": mu.std().item(),
        }
        info["logvar"] = {
            "shape": list(logvar.shape),
            "range": [logvar.min().item(), logvar.max().item()],
            "mean": logvar.mean().item(),
            "std": logvar.std().item(),
            "clamped_pct": (logvar_raw != logvar).float().mean().item() * 100,
        }

        # Warnings
        warnings = []
        if torch.isnan(mu).any():
            warnings.append("NaN in mu!")
        if torch.isnan(logvar).any():
            warnings.append("NaN in logvar!")
        if (logvar_raw > 10).any():
            warnings.append("Very large logvar (>10) — posterior may be collapsing")
        if (logvar_raw < -10).any():
            warnings.append("Very small logvar (<-10) — posterior may be overconfident")
        info["warnings"] = warnings

        return info


# ──────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LigandEncoder")
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pool-mode", choices=["max", "maxavg"], default="maxavg")
    parser.add_argument("--use-edge-conv", action="store_true", default=True)
    parser.add_argument("--no-edge-conv", action="store_false", dest="use_edge_conv")
    parser.add_argument("--edge-conv-k", type=int, default=8)
    parser.add_argument("--dynamic-graph", action="store_true", default=True)
    parser.add_argument("--no-dynamic-graph", action="store_false", dest="dynamic_graph")
    args = parser.parse_args()

    cfg = LigandEncoderConfig(
        max_atoms=args.max_atoms,
        latent_dim=args.latent_dim,
        pool_mode=args.pool_mode,
        use_edge_conv=args.use_edge_conv,
        edge_conv_k=args.edge_conv_k,
        dynamic_graph=args.dynamic_graph,
    )
    model = LigandEncoder(cfg)
    model.eval()

    # Dummy input — 2 molecules with different numbers of real atoms
    n_real = [20, 35]
    atom_coords = torch.randn(args.batch_size, args.max_atoms, 3)
    atom_types = torch.zeros(args.batch_size, args.max_atoms, dtype=torch.long)
    atom_mask = torch.zeros(args.batch_size, args.max_atoms, dtype=torch.bool)

    for i, n in enumerate(n_real):
        atom_types[i, :n] = torch.randint(0, 10, (n,))
        atom_mask[i, :n] = True

    # --- Parameter counts ---
    counts = model.param_counts()
    arch = "EdgeConv" if args.use_edge_conv else "PointNet"
    print(f"\n{'='*60}")
    print(f"  Architecture: {arch}  |  k={args.edge_conv_k}  |  pool={args.pool_mode}")
    print(f"{'='*60}")
    for name, count in counts.items():
        print(f"  {name:24s}  {count:>10,} params")
    print(f"{'='*60}")

    # --- Forward pass ---
    mu, logvar = model(atom_coords, atom_types, atom_mask)
    z = reparameterise(mu, logvar, deterministic=True)

    print(f"  Input:    B={args.batch_size}, N={args.max_atoms}, real_atoms={n_real}")
    print(f"  mu:       {list(mu.shape)}   range [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  logvar:   {list(logvar.shape)}   range [{logvar.min():.4f}, {logvar.max():.4f}]")
    print(f"  z:        {list(z.shape)}   range [{z.min():.4f}, {z.max():.4f}]")
    print(f"{'='*60}\n")

    # --- Debug info ---
    info = model.debug_forward(atom_coords, atom_types, atom_mask)
    for key, val in info.items():
        if isinstance(val, dict) and "shape" in val:
            line = f"  {key:24s}  shape={val['shape']}"
            if "range" in val:
                line += f"  range=[{val['range'][0]:.4f}, {val['range'][1]:.4f}]"
            if "mean" in val:
                line += f"  mean={val['mean']:.4f}  std={val['std']:.4f}"
            if "clamped_pct" in val:
                line += f"  clamped={val['clamped_pct']:.1f}%"
            if "graph" in val:
                line += f"  graph={val['graph']}  k={val['k']}"
            print(line)
        elif isinstance(val, list) and val:
            print(f"  {key:24s}  ⚠ {'; '.join(val)}")

    # --- Sample test ---
    z2, mu2, logvar2 = model.sample_z(atom_coords, atom_types, atom_mask)
    print(f"\n  z sample      shape={z2.shape}  range=[{z2.min():.3f}, {z2.max():.3f}]  "
          f"mean={z2.mean():.3f}  std={z2.std():.3f}")

    # --- Prior sample test ---
    z_prior = model.sample_from_prior(4, device=next(model.parameters()).device)
    print(f"  z prior       shape={z_prior.shape}  range=[{z_prior.min():.3f}, {z_prior.max():.3f}]  "
          f"mean={z_prior.mean():.3f}  std={z_prior.std():.3f}")

    # --- Architecture comparison ---
    print(f"\n--- Architecture comparison ---")
    for use_ec in [False, True]:
        for pm in ["max", "maxavg"]:
            cfg_cmp = LigandEncoderConfig(
                max_atoms=args.max_atoms,
                latent_dim=args.latent_dim,
                pool_mode=pm,
                use_edge_conv=use_ec,
                edge_conv_k=args.edge_conv_k,
            )
            m = LigandEncoder(cfg_cmp)
            total = sum(p.numel() for p in m.parameters())
            label = "EdgeConv" if use_ec else "PointNet"
            print(f"  {label:8s} + {pm:6s} pool  →  {total:>10,} params")

    print("\n✓ LigandEncoder test passed!")
