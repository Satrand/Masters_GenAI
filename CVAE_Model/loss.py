"""
loss.py — Loss function for the conditional VAE with Sinkhorn matching.

Components:
    1. Sinkhorn matching (training, GPU, differentiable) or Hungarian (eval, exact)
    2. Coordinate loss (Smooth L1) on matched pairs
    3. Atom type loss (Cross-entropy) on matched pairs
    4. Existence loss (BCE, class-balanced) for all slots
    5. KL divergence with free bits and configurable annealing schedule

§-marked parameters should be tuned for your dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass


# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

@dataclass
class LossConfig:
    """§ — Tune these loss weights for your task."""

    # --- Reconstruction weights ---
    lambda_coord: float = 1.0       # § coordinate loss weight
    lambda_type: float = 1.0        # § atom type classification weight
    lambda_exist: float = 5.0       # § existence prediction weight (was 1.0)

    # --- KL weight ---
    lambda_kl: float = 0.001        # § final KL weight; start low

    # --- Matching cost weights (control assignment, NOT loss magnitude) ---
    coord_cost_weight: float = 1.0  # § coord distance in cost matrix
    type_cost_weight: float = 1.0   # § type agreement in cost matrix

    # --- Cost matrix normalisation ---
    max_coord_range: float = 2.0    # § L1 distance range for coords (2.0 for [-1,1])
    #   Type cost auto-normalised by log(num_atom_types)

    # --- Coordinate loss type ---
    coord_loss_type: str = "smooth_l1"  # § "smooth_l1" or "mse"

    # --- Label smoothing for type loss ---
    type_label_smoothing: float = 0.0   # § 0.1 for mild regularisation

    # --- Existence loss class balance ---
    exist_pos_weight: str = "auto"  # § "auto" = compute from batch, float = fixed weight

    # --- KL free bits (per-dimension floor) ---
    free_bits_per_dim: float = 0.1  # § 0.0 = disabled; 0.1-0.5 typical

    # --- KL annealing ---
    kl_schedule: str = "linear"     # § "linear", "sigmoid", "cyclical"
    kl_anneal_steps: int = 10000    # § steps to reach full KL weight
    kl_cyclical_cycles: int = 4     # § cycles for cyclical schedule
    kl_cyclical_prop: float = 0.5   # § proportion of each cycle with full weight

    # --- Matching mode ---
    use_sinkhorn: bool = True       # § True = Sinkhorn (train), False = Hungarian always
    sinkhorn_iters: int = 20        # § Sinkhorn iterations
    sinkhorn_temp: float = 1.0      # § Sinkhorn temperature (lower = sharper)

    # --- Coordinate repulsion (prevents slot collapse) ---
    repulsion_weight: float = 2.0   # § weight for pairwise repulsion loss
    repulsion_min_dist: float = 0.15  # § in normalised coord space (~2.4 Å if centre_scale)


# ──────────────────────────────────────────────
#  KL Annealer
# ──────────────────────────────────────────────

class KLAnnealer:
    """KL weight scheduler with linear, sigmoid, or cyclical schedules."""

    def __init__(self, config: LossConfig):
        self.target = config.lambda_kl
        self.anneal_steps = config.kl_anneal_steps
        self.schedule = config.kl_schedule
        self.cycles = config.kl_cyclical_cycles
        self.prop = config.kl_cyclical_prop
        self.total_steps = 0

    def get_weight(self, step):
        if self.anneal_steps <= 0:
            return self.target

        if self.schedule == "linear":
            progress = min(step / self.anneal_steps, 1.0)
            return self.target * progress

        elif self.schedule == "sigmoid":
            k = 10.0
            midpoint = self.anneal_steps / 2.0
            x = k * (step - midpoint) / max(midpoint, 1.0)
            sigmoid = 1.0 / (1.0 + math.exp(-x))
            return self.target * sigmoid

        elif self.schedule == "cyclical":
            # Cycle across the ENTIRE training run, not just anneal_steps
            horizon = self.total_steps if self.total_steps > 0 else self.anneal_steps
            if horizon <= 0:
                return self.target
            
            cycle_len = horizon / self.cycles
            position_in_cycle = (step % cycle_len) / cycle_len  # [0, 1)
            
            # prop = fraction of cycle at FULL weight
            # ramp for (1 - prop), hold for prop
            if position_in_cycle < (1.0 - self.prop):
                ramp_progress = position_in_cycle / (1.0 - self.prop)
                return self.target * ramp_progress
            else:
                return self.target

        else:
            raise ValueError(f"Unknown KL schedule: {self.schedule}")

    def get_progress(self, step):
        """Return annealing progress 0→1 (for logging)."""
        if self.anneal_steps <= 0:
            return 1.0
        return min(step / self.anneal_steps, 1.0)


# ──────────────────────────────────────────────
#  Sinkhorn Matcher (GPU, differentiable)
# ──────────────────────────────────────────────

class SinkhornMatcher:
    """Batched Sinkhorn matching — differentiable approximation to
    the optimal assignment problem.

    Handles rectangular matrices (N_pred ≥ N_real) by padding to square
    with dummy columns at high cost. This ensures the doubly-stochastic
    constraint is mathematically consistent.
    """

    def __init__(self, iters: int = 20, temp: float = 1.0):
        self.iters = iters
        self.temp = temp

    def match(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_matrix: (B, N_pred, N_real) — lower = better match.
                         Padding positions should be set to a large
                         finite value (e.g. 1e8) to avoid NaN.

        Returns:
            assignment: (B, N_pred, N_real) — soft assignment.
                        Each real target column sums to 1.
                        Each prediction row sums to ≤ 1 (mass can go to dummy).
        """
        B, N, M = cost_matrix.shape

        # ── Pad to square to fix marginal inconsistency ──
        # N_pred (slots) ≥ N_real (targets) always in our pipeline.
        # We pad dummy columns with high cost so excess predictions
        # are absorbed harmlessly. After Sinkhorn, extract real columns.
        if N != M:
            padded_cost = torch.full(
                (B, N, N), 1e8,
                device=cost_matrix.device,
                dtype=cost_matrix.dtype,
            )
            padded_cost[:, :, :M] = cost_matrix
            log_alpha = -padded_cost / self.temp

            # Standard Sinkhorn on square matrix
            for _ in range(self.iters):
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

            # Extract assignment to real targets only.
            # Because the square matrix is doubly-stochastic:
            #   - Each real target column still sums to 1.0
            #   - Each prediction row sums to ≤ 1.0 (remainder went to dummy)
            assignment = log_alpha.exp()[:, :, :M]

        else:
            # Already square — standard Sinkhorn
            log_alpha = -cost_matrix / self.temp
            for _ in range(self.iters):
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            assignment = log_alpha.exp()

        return assignment
# ──────────────────────────────────────────────
#  Hungarian Matcher (CPU, exact — eval only)
# ──────────────────────────────────────────────

def hungarian_match_single(
    pred_coords: torch.Tensor,
    pred_type_logits: torch.Tensor,
    target_coords: torch.Tensor,
    target_types: torch.Tensor,
    coord_cost_weight: float = 1.0,
    type_cost_weight: float = 1.0,
    max_coord_range: float = 2.0,
) -> tuple:
    """Hungarian matching for a SINGLE sample (no batch dimension).

    Args:
        pred_coords:      (N_pred, 3)
        pred_type_logits: (N_pred, K)
        target_coords:    (N_real, 3)
        target_types:     (N_real,)
        coord_cost_weight, type_cost_weight: matching cost weights
        max_coord_range:  L1 distance normalisation

    Returns:
        pred_indices:  (N_real,) — which predicted slot each target maps to
        target_indices: (N_real,) — always 0..N_real-1
    """
    K = pred_type_logits.shape[-1]

    # Coordinate cost
    coord_cost = torch.cdist(
        pred_coords.unsqueeze(0), target_coords.unsqueeze(0), p=1
    ).squeeze(0) / max_coord_range

    # Type cost
    type_logprobs = F.log_softmax(pred_type_logits, dim=-1)
    type_cost = -type_logprobs[:, target_types] / math.log(K)

    # Combined
    cost = coord_cost_weight * coord_cost + type_cost_weight * type_cost

    # Hungarian (CPU)
    cost_np = cost.detach().cpu().numpy()
    pred_indices, target_indices = linear_sum_assignment(cost_np)

    return pred_indices, target_indices


# ──────────────────────────────────────────────
#  CVaE Loss
# ──────────────────────────────────────────────

class CVaELoss(nn.Module):
    """
    Conditional VAE loss with Sinkhorn (training) / Hungarian (eval) matching.

    Training path (Sinkhorn):
        - Batched cost matrix construction on GPU
        - Differentiable soft assignment via Sinkhorn iterations
        - Weighted coord/type loss on soft assignment
        - Soft existence targets from assignment mass

    Eval path (Hungarian):
        - Per-sample exact matching on CPU
        - Hard coord/type loss on matched pairs
        - Hard existence targets (1 for matched, 0 for unmatched)
    """

    def __init__(self, config: LossConfig = None, num_atom_types: int = 13):
        super().__init__()
        self.cfg = config or LossConfig()
        self.annealer = KLAnnealer(self.cfg)
        self.sinkhorn = SinkhornMatcher(
            iters=self.cfg.sinkhorn_iters,
            temp=self.cfg.sinkhorn_temp,
        )
        self.num_atom_types = num_atom_types

    def _build_real_targets(
        self,
        target_coords: torch.Tensor,
        target_types: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple:
        """Extract real (non-padding) targets from padded tensors.

        Returns:
            tgt_coords_real: (B, max_n_real, 3)
            tgt_types_real:  (B, max_n_real) long
            n_real_per_sample: (B,) long
            real_mask:       (B, max_n_real) bool — True for real positions
        """
        B, N = target_mask.shape
        device = target_mask.device

        n_real_per_sample = target_mask.sum(dim=1).long()  # (B,)
        max_n_real = n_real_per_sample.max().item()

        # Handle edge case: no real atoms in entire batch
        if max_n_real == 0:
            tgt_coords_real = torch.zeros(B, 1, 3, device=device)
            tgt_types_real = torch.zeros(B, 1, dtype=torch.long, device=device)
            real_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            return tgt_coords_real, tgt_types_real, n_real_per_sample, real_mask

        tgt_coords_real = torch.zeros(B, max_n_real, 3, device=device)
        tgt_types_real = torch.zeros(B, max_n_real, dtype=torch.long, device=device)

        for b in range(B):
            nr = n_real_per_sample[b].item()
            if nr > 0:
                idx = target_mask[b].nonzero(as_tuple=True)[0][:nr]
                tgt_coords_real[b, :nr] = target_coords[b, idx]
                tgt_types_real[b, :nr] = target_types[b, idx]

        # Mask indicating which columns are real (vs padding)
        real_mask = torch.arange(max_n_real, device=device).unsqueeze(0) < n_real_per_sample.unsqueeze(1)  # (B, max_n_real)

        return tgt_coords_real, tgt_types_real, n_real_per_sample, real_mask

    def _build_cost_matrix(
        self,
        pred_coords: torch.Tensor,
        pred_type_logits: torch.Tensor,
        tgt_coords_real: torch.Tensor,
        tgt_types_real: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalised cost matrix (batched, on GPU).

        Args:
            pred_coords:      (B, N, 3)
            pred_type_logits: (B, N, K)
            tgt_coords_real:  (B, max_n_real, 3)
            tgt_types_real:   (B, max_n_real) long
            real_mask:        (B, max_n_real) bool

        Returns:
            cost_matrix: (B, N, max_n_real) — large finite value for padding
        """
        B, N, K = pred_type_logits.shape
        max_n_real = tgt_coords_real.shape[1]

        # Coordinate cost: L1 distance, normalised
        coord_cost = torch.cdist(pred_coords, tgt_coords_real, p=1)  # (B, N, max_n_real)
        coord_cost = coord_cost / self.cfg.max_coord_range

        # Type cost: negative log-prob of correct type, normalised
        type_logprobs = F.log_softmax(pred_type_logits, dim=-1)  # (B, N, K)
        tgt_types_exp = tgt_types_real.unsqueeze(1).expand(-1, N, -1)  # (B, N, max_n_real)
        type_neglogprob = -torch.gather(type_logprobs, 2, tgt_types_exp)  # (B, N, max_n_real)
        type_cost = type_neglogprob / math.log(K)

        # Combined cost
        cost_matrix = (
            self.cfg.coord_cost_weight * coord_cost +
            self.cfg.type_cost_weight * type_cost
        )

        # Mask padding target positions with large finite value (NOT inf)
        # Using inf causes NaN in Sinkhorn; large finite makes them very unlikely
        cost_matrix = cost_matrix.masked_fill(~real_mask.unsqueeze(1), 1e8)

        return cost_matrix, type_neglogprob, coord_cost

    
    def _hungarian_forward(
        self,
        pred_coords: torch.Tensor,
        pred_type_logits: torch.Tensor,
        pred_exist_logits: torch.Tensor,
        target_coords: torch.Tensor,
        target_types: torch.Tensor,
        target_mask: torch.Tensor,
        n_real_per_sample: torch.Tensor,
    ) -> tuple:
        """Eval path: Hungarian exact matching → hard losses + metrics."""

        B, N, K = pred_type_logits.shape
        device = pred_coords.device

        coord_losses = []
        type_losses = []
        type_accuracies = []
        coord_dists = []
        exist_target = torch.zeros(B, N, device=device)

        for b in range(B):
            nr = n_real_per_sample[b].item()
            if nr == 0:
                continue

            # Extract real targets
            real_idx = target_mask[b].nonzero(as_tuple=True)[0][:nr]
            t_coords = target_coords[b, real_idx]     # (nr, 3)
            t_types = target_types[b, real_idx]        # (nr,)

            # Hungarian matching
            pred_idx, tgt_idx = hungarian_match_single(
                pred_coords[b], pred_type_logits[b],
                t_coords, t_types,
                self.cfg.coord_cost_weight,
                self.cfg.type_cost_weight,
                self.cfg.max_coord_range,
            )

            # Coordinate loss
            matched_pred = pred_coords[b, pred_idx]       # (nr, 3)
            matched_tgt = t_coords[tgt_idx]                # (nr, 3)

            if self.cfg.coord_loss_type == "smooth_l1":
                c_loss = F.smooth_l1_loss(matched_pred, matched_tgt, reduction="mean")
            else:
                c_loss = F.mse_loss(matched_pred, matched_tgt, reduction="mean")
            coord_losses.append(c_loss)

            # Type loss
            matched_logits = pred_type_logits[b, pred_idx]  # (nr, K)
            if self.cfg.type_label_smoothing > 0:
                t_loss = F.cross_entropy(
                    matched_logits, t_types[tgt_idx],
                    label_smoothing=self.cfg.type_label_smoothing,
                    reduction="mean",
                )
            else:
                t_loss = F.cross_entropy(matched_logits, t_types[tgt_idx], reduction="mean")
            type_losses.append(t_loss)

            # ── Metrics for this sample ──
            pred_types = pred_type_logits[b].argmax(dim=-1)
            type_correct = (pred_types[pred_idx] == t_types[tgt_idx]).float().mean().item()
            type_accuracies.append(type_correct)

            dists = torch.norm(matched_pred - matched_tgt, dim=-1)
            coord_dists.append(dists.mean().item())

            # Mark matched slots as existing
            exist_target[b, pred_idx] = 1.0

        n_valid = len(coord_losses)
        if n_valid > 0:
            coord_loss = torch.stack(coord_losses).mean()
            type_loss = torch.stack(type_losses).mean()
            match_coord_dist = float(np.mean(coord_dists))
            match_type_acc = float(np.mean(type_accuracies))
        else:
            coord_loss = torch.tensor(0.0, device=device)
            type_loss = torch.tensor(0.0, device=device)
            match_coord_dist = 0.0
            match_type_acc = 0.0

        return coord_loss, type_loss, exist_target, n_valid, match_coord_dist, match_type_acc
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        pred_type_logits: torch.Tensor,
        pred_exist_logits: torch.Tensor,
        target_coords: torch.Tensor,
        target_types: torch.Tensor,
        target_mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        step: int = 0,
    ) -> tuple:
        """
        Args:
            pred_coords:       (B, N, 3)   float — predicted atom coordinates
            pred_type_logits:  (B, N, K)   float — predicted atom type logits
            pred_exist_logits: (B, N, 1)   float — predicted existence logit
            target_coords:     (B, N, 3)   float — target coordinates (padded)
            target_types:      (B, N)      long  — target type indices (padded)
            target_mask:       (B, N)      bool  — True = real atom
            mu:                (B, z_dim)  float — ligand encoder mu
            logvar:            (B, z_dim)  float — ligand encoder logvar
            step:              int         — training step (for KL annealing)

        Returns:
            total_loss: scalar
            loss_dict:  dict of individual components for logging
        """
        B, N, K = pred_type_logits.shape
        device = pred_coords.device

        # Safety check
        max_real = target_mask.sum(dim=1).max().item()
        assert max_real <= N, (
            f"More real atoms ({max_real}) than prediction slots ({N}). "
            f"Check max_atoms in DataConfig vs DecoderConfig."
        )

        # ── Extract real targets ──
        tgt_coords_real, tgt_types_real, n_real_per_sample, real_mask = \
            self._build_real_targets(target_coords, target_types, target_mask)

        # ── Choose matching path ──
        use_sinkhorn = self.cfg.use_sinkhorn and self.training

        if use_sinkhorn:
            # ── Sinkhorn path (batched, GPU, differentiable) ──
            cost_matrix, type_neglogprob, coord_cost_raw = self._build_cost_matrix(
                pred_coords, pred_type_logits,
                tgt_coords_real, tgt_types_real, real_mask,
            )

            # Soft assignment
            assignment = self.sinkhorn.match(cost_matrix)  # (B, N, max_n_real)

            # Zero out assignment to padding targets
            assignment = assignment * real_mask.unsqueeze(1).float()

            # Coordinate loss (weighted by assignment)
            coord_diff = pred_coords.unsqueeze(2) - tgt_coords_real.unsqueeze(1)  # (B, N, max_n_real, 3)

            if self.cfg.coord_loss_type == "smooth_l1":
                coord_loss_per_pair = F.smooth_l1_loss(
                    coord_diff, torch.zeros_like(coord_diff), reduction='none'
                ).sum(dim=-1)  # (B, N, max_n_real)
            else:
                coord_loss_per_pair = (coord_diff ** 2).sum(dim=-1)

            coord_loss_per_sample = (coord_loss_per_pair * assignment).sum(dim=(1, 2))
            coord_loss_per_sample = coord_loss_per_sample / n_real_per_sample.float().clamp(min=1)
            coord_loss = coord_loss_per_sample.mean()

            # Type loss (weighted by assignment, with optional label smoothing)
            if self.cfg.type_label_smoothing > 0:
                eps = self.cfg.type_label_smoothing
                log_probs = F.log_softmax(pred_type_logits, dim=-1)  # (B, N, K)
                mean_log_probs = log_probs.mean(dim=-1, keepdim=True)  # (B, N, 1)
                # Smoothed CE: (1-eps) * hard_CE + eps * (-mean_log_probs)
                type_loss_per_pair = (1.0 - eps) * type_neglogprob - eps * mean_log_probs
            else:
                type_loss_per_pair = type_neglogprob

            type_loss_per_sample = (type_loss_per_pair * assignment).sum(dim=(1, 2))
            type_loss_per_sample = type_loss_per_sample / n_real_per_sample.float().clamp(min=1)
            type_loss = type_loss_per_sample.mean()

            # ── Existence target: HARD threshold (was soft mass) ──
            # This forces each slot to make a binary decision.
            exist_target = (assignment.sum(dim=2) > 0.5).float()  # (B, N)

        else:
            # ── Hungarian path (per-sample, CPU, exact) ──
            coord_loss, type_loss, exist_target, n_valid, match_coord_dist, match_type_acc = self._hungarian_forward(
                pred_coords, pred_type_logits, pred_exist_logits,
                target_coords, target_types, target_mask,
                n_real_per_sample,
            )

        # ── Coordinate repulsion (prevents slot collapse) ──
        if self.training and N > 1:
            # Pairwise L2 distances between predicted coords
            dist_matrix = torch.cdist(pred_coords, pred_coords, p=2)  # (B, N, N)
            # Mask diagonal
            eye = torch.eye(N, device=device).unsqueeze(0)
            dist_matrix = dist_matrix + eye * 1e6
            # Penalise any pair closer than threshold
            repulsion = F.relu(self.cfg.repulsion_min_dist - dist_matrix).pow(2)
            repulsion = repulsion.mean()
        else:
            repulsion = torch.tensor(0.0, device=device)

        # ── Existence loss (class-balanced BCE) ──
        if self.cfg.exist_pos_weight == "auto":
            with torch.no_grad():
                n_pos = exist_target.sum().clamp(min=1)
                n_neg = exist_target.numel() - n_pos
                pos_weight = (n_neg / n_pos).clamp(min=0.5, max=10.0)
                pos_weight = pos_weight.to(pred_exist_logits.device)
        else:
            pos_weight = torch.tensor(
                float(self.cfg.exist_pos_weight),
                device=pred_exist_logits.device
            )

        exist_loss = F.binary_cross_entropy_with_logits(
            pred_exist_logits.squeeze(-1),  # (B, N)
            exist_target,                    # (B, N)
            pos_weight=pos_weight,
            reduction="mean",
        )

        # ── Matching quality metrics (for logging, no grad) ──
        with torch.no_grad():
            if use_sinkhorn:
                # Approximate hard assignment from soft for metrics
                hard_assignment = (assignment.sum(dim=2) > 0.5).float()

                # Mean coord distance (weighted by assignment)
                matched_coord_dist = (coord_loss_per_pair.detach() * assignment.detach()).sum(dim=(1, 2))
                assignment_mass = assignment.detach().sum(dim=(1, 2)).clamp(min=1)
                match_coord_dist = (matched_coord_dist / assignment_mass).mean().item()

                # Type accuracy (fraction of correct top-1 predictions for assigned slots)
                pred_types = pred_type_logits.detach().argmax(dim=-1)  # (B, N)
                # For each real target, which pred slot has highest assignment?
                best_pred = assignment.detach().argmax(dim=1)  # (B, max_n_real)
                best_pred_types = torch.gather(pred_types, 1, best_pred)  # (B, max_n_real)
                correct = (best_pred_types == tgt_types_real).float() * real_mask.float()
                match_type_acc = correct.sum() / real_mask.float().sum().clamp(min=1).item()
                match_type_acc = match_type_acc.item() if isinstance(match_type_acc, torch.Tensor) else match_type_acc
            

        # ── KL divergence with free bits ──
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z_dim)
        kl_per_dim = torch.clamp(kl_per_dim, min=self.cfg.free_bits_per_dim)
        kl_per_sample = kl_per_dim.sum(dim=-1)  # (B,)
        kl_loss = kl_per_sample.mean()

        # ── KL annealing ──
        kl_weight = self.annealer.get_weight(step)

        # ── Total loss ──
        total_loss = (
            self.cfg.lambda_coord * coord_loss +
            self.cfg.lambda_type * type_loss +
            self.cfg.lambda_exist * exist_loss +
            kl_weight * kl_loss +
            self.cfg.repulsion_weight * repulsion  # ← NEW
        )

        # ── Logging ──
        loss_dict = {
            "total": total_loss.item(),
            "coord": coord_loss.item(),
            "type": type_loss.item(),
            "exist": exist_loss.item(),
            "kl": kl_loss.item(),
            "kl_weight": kl_weight,
            "kl_free_bits": self.cfg.free_bits_per_dim,
            "pos_weight": pos_weight if isinstance(pos_weight, float) else pos_weight.item(),
            "match_coord_dist": match_coord_dist,
            "match_type_acc": match_type_acc,
            "n_real_mean": n_real_per_sample.float().mean().item(),
            "repulsion": repulsion.item(),  # ← NEW
        }

        return total_loss, loss_dict

    # ── Debug: inspect matching quality ──

    @torch.no_grad()
    def debug_match(
        self,
        pred_coords: torch.Tensor,
        pred_type_logits: torch.Tensor,
        pred_exist_logits: torch.Tensor,
        target_coords: torch.Tensor,
        target_types: torch.Tensor,
        target_mask: torch.Tensor,
        sample_idx: int = 0,
    ) -> dict:
        """Detailed matching info for one sample. Uses Hungarian (exact).

        Returns:
            dict with matched pairs, coord distances, type accuracy,
            existence probs, and unmatched slot info.
        """
        b = sample_idx
        n_real = target_mask[b].sum().item()
        real_idx = target_mask[b].nonzero(as_tuple=True)[0][:n_real]
        t_coords = target_coords[b, real_idx]
        t_types = target_types[b, real_idx]

        # Hungarian matching
        pred_idx, tgt_idx = hungarian_match_single(
            pred_coords[b], pred_type_logits[b],
            t_coords, t_types,
            self.cfg.coord_cost_weight,
            self.cfg.type_cost_weight,
            self.cfg.max_coord_range,
        )

        # Predicted types and existence probs
        pred_types = pred_type_logits[b].argmax(dim=-1)
        exist_probs = torch.sigmoid(pred_exist_logits[b].squeeze(-1))

        # Build results
        pairs = []
        coord_dists = []
        type_correct = 0

        for i in range(len(pred_idx)):
            pi = pred_idx[i]
            ti = tgt_idx[i]
            dist = (pred_coords[b, pi] - t_coords[ti]).norm().item()
            type_match = pred_types[pi].item() == t_types[ti].item()
            if type_match:
                type_correct += 1
            coord_dists.append(dist)
            pairs.append({
                "pred_slot": pi,
                "pred_type": pred_types[pi].item(),
                "target_type": t_types[ti].item(),
                "coord_dist": dist,
                "type_correct": type_match,
                "exist_prob": exist_probs[pi].item(),
            })

        matched_slots = set(pred_idx.tolist())
        unmatched = [i for i in range(pred_coords.shape[1]) if i not in matched_slots]

        # Existence stats
        matched_exist_probs = exist_probs[list(matched_slots)].mean().item() if matched_slots else 0.0
        unmatched_exist_probs = exist_probs[unmatched].mean().item() if unmatched else 0.0

        result = {
            "sample_idx": b,
            "n_real": n_real,
            "n_matched": len(pred_idx),
            "type_accuracy": type_correct / max(len(pred_idx), 1),
            "mean_coord_dist": np.mean(coord_dists) if coord_dists else 0.0,
            "matched_exist_prob_mean": matched_exist_probs,
            "unmatched_exist_prob_mean": unmatched_exist_probs,
            "pairs": pairs,
            "unmatched_slots": unmatched,
        }

        return result


# ──────────────────────────────────────────────
#  CLI test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CVaELoss")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=88)
    parser.add_argument("--num-types", type=int, default=13)
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()

    B, N, K = args.batch_size, args.max_atoms, args.num_types
    device = "cpu"

    cfg = LossConfig(lambda_kl=0.001, kl_anneal_steps=5000)
    loss_fn = CVaELoss(cfg, num_atom_types=K)

    # --- Simulate predictions ---
    pred_coords = torch.randn(B, N, 3).tanh() * 0.5
    pred_type_logits = torch.randn(B, N, K)
    pred_exist_logits = torch.randn(B, N, 1)

    # --- Simulate targets ---
    target_coords = torch.randn(B, N, 3).tanh() * 0.5
    target_types = torch.zeros(B, N, dtype=torch.long)
    target_mask = torch.zeros(B, N, dtype=torch.bool)

    n_atoms_per_sample = [20, 35, 50, 10]
    for b, n in enumerate(n_atoms_per_sample):
        target_mask[b, :n] = True
        target_types[b, :n] = torch.randint(1, K - 1, (n,))

    # --- Simulate encoder outputs ---
    mu = torch.randn(B, args.latent_dim) * 0.1
    logvar = torch.randn(B, args.latent_dim) * 0.1 - 1

    # --- Test different KL schedules ---
    print("=== CVaELoss Test ===\n")
    print(f"  Batch: {B} samples, {N} max atoms, {K} types")
    print(f"  Real atoms per sample: {n_atoms_per_sample}\n")

    for schedule in ["linear", "sigmoid", "cyclical"]:
        cfg_s = LossConfig(
            lambda_kl=0.01,
            kl_anneal_steps=10000,
            kl_schedule=schedule,
            kl_cyclical_cycles=4,
        )
        loss_fn_s = CVaELoss(cfg_s, num_atom_types=K)
        print(f"  --- KL schedule: {schedule} ---")
        for step in [0, 1000, 2500, 5000, 7500, 10000, 15000]:
            w = loss_fn_s.annealer.get_weight(step)
            print(f"    Step {step:5d}  →  kl_weight={w:.5f}")

    # --- Test Sinkhorn vs Hungarian ---
    print(f"\n  --- Sinkhorn vs Hungarian ---")
    for use_sink in [True, False]:
        cfg_m = LossConfig(use_sinkhorn=use_sink)
        loss_fn_m = CVaELoss(cfg_m, num_atom_types=K)
        if not use_sink:
            loss_fn_m.eval()

        total, d = loss_fn_m(
            pred_coords, pred_type_logits, pred_exist_logits,
            target_coords, target_types, target_mask,
            mu, logvar, step=5000,
        )
        mode = "Sinkhorn" if use_sink else "Hungarian"
        print(f"    {mode:10s}  total={d['total']:.4f}  "
              f"coord={d['coord']:.4f}  type={d['type']:.4f}  "
              f"exist={d['exist']:.4f}  kl={d['kl']:.4f}  "
              f"match_dist={d['match_coord_dist']:.4f}  "
              f"type_acc={d['match_type_acc']:.3f}  "
              f"repulsion={d.get('repulsion', 0):.4f}")

    # --- Test free bits ---
    print(f"\n  --- Free bits ---")
    for fb in [0.0, 0.1, 0.5]:
        cfg_fb = LossConfig(free_bits_per_dim=fb, lambda_kl=0.01)
        loss_fn_fb = CVaELoss(cfg_fb, num_atom_types=K)
        total, d = loss_fn_fb(
            pred_coords, pred_type_logits, pred_exist_logits,
            target_coords, target_types, target_mask,
            mu, logvar, step=5000,
        )
        print(f"    free_bits={fb:.1f}  kl={d['kl']:.4f}  total={d['total']:.4f}")

    # --- Test edge case: sample with 0 real atoms ---
    print(f"\n  --- Edge case: 0 real atoms ---")
    edge_mask = torch.zeros(B, N, dtype=torch.bool)
    edge_mask[0, :10] = True  # Only first sample has atoms
    edge_types = torch.zeros(B, N, dtype=torch.long)
    edge_types[0, :10] = torch.randint(1, K - 1, (10,))
    edge_coords = torch.randn(B, N, 3).tanh() * 0.5

    loss_fn_edge = CVaELoss(LossConfig(), num_atom_types=K)
    total, d = loss_fn_edge(
        pred_coords, pred_type_logits, pred_exist_logits,
        edge_coords, edge_types, edge_mask,
        mu, logvar, step=1000,
    )
    print(f"    total={d['total']:.4f}  coord={d['coord']:.4f}  "
          f"type={d['type']:.4f}  exist={d['exist']:.4f}  kl={d['kl']:.4f}")
    print(f"    (3 samples with 0 atoms should contribute 0 to coord/type loss)")

    # --- Test debug_match ---
    print(f"\n  --- Debug Match (sample 0) ---")
    loss_fn_dbg = CVaELoss(LossConfig(), num_atom_types=K)
    result = loss_fn_dbg.debug_match(
        pred_coords, pred_type_logits, pred_exist_logits,
        target_coords, target_types, target_mask,
        sample_idx=0,
    )
    print(f"    n_real={result['n_real']}, n_matched={result['n_matched']}")
    print(f"    type_accuracy={result['type_accuracy']:.3f}")
    print(f"    mean_coord_dist={result['mean_coord_dist']:.4f}")
    print(f"    matched_exist_prob={result['matched_exist_prob_mean']:.3f}")
    print(f"    unmatched_exist_prob={result['unmatched_exist_prob_mean']:.3f}")
    if result['pairs']:
        print(f"    First 5 pairs:")
        for p in result['pairs'][:5]:
            print(f"      slot={p['pred_slot']:3d}  pred_type={p['pred_type']:2d}  "
                  f"tgt_type={p['target_type']:2d}  dist={p['coord_dist']:.4f}  "
                  f"correct={p['type_correct']}  exist={p['exist_prob']:.3f}")

    # --- Sanity: perfect predictions ---
    print(f"\n  --- Sanity: perfect predictions ---")
    perfect_coords = target_coords.clone()
    perfect_type_logits = torch.zeros(B, N, K)
    perfect_exist = target_mask.float().unsqueeze(-1) * 10 - 5
    for b in range(B):
        for i in range(N):
            if target_mask[b, i]:
                perfect_type_logits[b, i, target_types[b, i]] = 10.0

    total, d = loss_fn(
        perfect_coords, perfect_type_logits, perfect_exist,
        target_coords, target_types, target_mask,
        mu, logvar, step=0,
    )
    print(f"    total={d['total']:.4f}  coord={d['coord']:.4f}  "
          f"type={d['type']:.4f}  exist={d['exist']:.4f}  kl={d['kl']:.4f}")
    print("    (coord+type+exist should be ~0, kl will be non-zero)")

    # --- Annealer progress ---
    print(f"\n  --- KL Annealer ---")
    annealer = KLAnnealer(cfg)
    for step in [0, 1000, 2500, 5000, 7500, 10000]:
        print(f"    Step {step:5d}  →  kl_weight={annealer.get_weight(step):.5f}  "
              f"({annealer.get_progress(step)*100:.0f}%)")

    print("\n✓ CVaELoss test passed!")
