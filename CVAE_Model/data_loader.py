"""
data.py — Dataset and DataLoader for conditional VAE: multi-channel MIF (.cmap) → ligand PDB

Input:  .cmap (h5py) containing 5 separate 3D grids (MIF fields)
Output: .pdb ligand structure → atom coords, types, and mask

Variable grid sizes are handled by centre-padding or centre-cropping to a
uniform target shape. Since MIF data is centred to correspond to PDB
coordinates, symmetric padding with zeros (no MIF signal) preserves alignment.

All parameters that must be updated for your dataset are marked with: §

Test with: python data.py --cmap file.cmap --pdb matched.pdb
"""

import os
import glob
import argparse
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass, field


# ============================================================
# Configuration — update all §-marked fields for your dataset
# ============================================================

@dataclass
class DataConfig:
    """Single place for all dataset-dependent parameters."""

    # ---- Dataset settings ----
    dataset_dir: Optional[str] = None

    # § Path to dataset(s) inside the h5py file.
    dataset_path: Optional[Union[str, List[str]]] = None

    num_channels: int = 5
    expected_grid_shape: Tuple[int, int, int] = (128, 128, 128)
    mif_norm: str = "none"
    mif_global_max: float = 10.0

    atom_types: List[str] = field(default_factory=lambda: [
        "H", "C", "N", "O", "F", "S", "Cl", "Br", "P", "I", "B",
    ])
    max_atoms: int = 96
    voxel_spacing: float = 0.25
    grid_origin: Tuple[float, float, float] = field(init=False)
    coord_norm: str = "none"

    batch_size: int = 16
    num_workers: int = 4
    val_fraction: float = 0.1
    seed: int = 42
    cache: bool = False
    pin_memory: bool = True

    pad_id: int = field(init=False)
    unk_id: int = field(init=False)
    num_atom_types: int = field(init=False)
    elem_to_id: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.pad_id = 0
        self.elem_to_id = {elem: i + 1 for i, elem in enumerate(self.atom_types)}
        self.unk_id = len(self.atom_types) + 1
        self.num_atom_types = len(self.atom_types) + 2
        self.grid_origin = tuple(
            -(np.array(self.expected_grid_shape, dtype=np.float64) * self.voxel_spacing) / 2.0
        )

    # ──────────────────────────────────────────────
    #  Single source of truth → model config
    # ──────────────────────────────────────────────

    def to_model_config(self):
        """Build a CVaeConfig with all data-dependent fields synced from this DataConfig.

        Architecture fields (base_width, hidden_dim, etc.) keep their defaults.
        Override them on the returned config if needed.

        Returns:
            CVaeConfig instance
        """
        from model import CVaeConfig

        return CVaeConfig(
            # Data-dependent (REQUIRED — no defaults in CVaeConfig)
            input_shape=self.expected_grid_shape,
            in_channels=self.num_channels,
            num_atom_types=self.num_atom_types,
            max_atoms=self.max_atoms,
            padding_idx=self.pad_id,
            # Architecture fields NOT set here — keep their defaults
        )

    def summary(self) -> str:
        """Human-readable summary of the config."""
        lines = [
            "DataConfig summary:",
            f"  Channels:       {self.num_channels}",
            f"  Grid shape:     {self.expected_grid_shape}",
            f"  MIF norm:       {self.mif_norm}"
            + (f" (global_max={self.mif_global_max})" if self.mif_norm == "global" else ""),
            f"  Coord norm:     {self.coord_norm}",
            f"  Max atoms:      {self.max_atoms}",
            f"  Atom types:     {self.num_atom_types}  (pad=0, "
            f"elements=1..{len(self.atom_types)}, unk={self.unk_id})",
            f"  Vocab:          {self.atom_types}",
            f"  Batch size:     {self.batch_size}",
            f"  Cache:          {self.cache}",
            f"  Voxel spacing:  {self.voxel_spacing}",
            f"  Grid origin:    {self.grid_origin}  (auto: -(size*step)/2)",
        ]
        return "\n".join(lines)


# ============================================================
# Grid padding / cropping
# ============================================================

def centre_pad_or_crop(
    grid: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Pad or crop a 3D grid to target_shape, keeping the centre fixed.

    For grids smaller than target: symmetric zero-padding on both sides.
    For grids larger than target: symmetric centre-cropping on both sides.
    For grids matching target: returned as-is (no copy).

    Because MIF data is centred to correspond to PDB coordinates,
    symmetric padding with zeros (= no MIF signal) preserves alignment
    between the grid and the atom positions.

    Args:
        grid:         (..., D, H, W) float32
        target_shape: (D, H, W) desired shape
        pad_value:    fill value for padding (0.0 = no MIF signal)

    Returns:
        (..., D, H, W) float32 with shape == target_shape,
        shift: how many voxels the origin moved in each dimension
    """
    target_shape = tuple(target_shape)
    
    spatial = grid.shape[-3:]
    if spatial == target_shape:
        return grid, (0, 0, 0)

    result_shape = grid.shape[:-3] + target_shape
    result = np.full(result_shape, pad_value, dtype=grid.dtype)

    src_slices = []
    dst_slices = []
    shifts = []

    for src_dim, tgt_dim in zip(spatial, target_shape):
        if src_dim <= tgt_dim:
            # Smaller than target: centre with padding on both sides
            offset = (tgt_dim - src_dim) // 2
            src_slices.append(slice(None))
            dst_slices.append(slice(offset, offset + src_dim))
            shifts.append(-offset)
        else:
            # Larger than target: crop from centre
            offset = (src_dim - tgt_dim) // 2
            src_slices.append(slice(offset, offset + tgt_dim))
            dst_slices.append(slice(None))
            shifts.append(offset)

    result[(..., *dst_slices)] = grid[(..., *src_slices)]
    return result, tuple(shifts)


# ============================================================
# MIF loading and normalisation
# ============================================================

def _read_step_attr(f: h5py.File, dataset_path: str) -> Optional[np.ndarray]:
    """Read step from the dataset's parent group, falling back to root."""
    step = None
    if "/" in dataset_path:
        parent = dataset_path.rsplit("/", 1)[0]
        if parent in f:
            step = f[parent].attrs.get("step", None)
    if step is None:
        step = f.attrs.get("step", None)
    if step is not None:
        step = np.asarray(step, dtype=np.float64)
    return step


def _read_file_origin_attr(f: h5py.File, dataset_path: str) -> Optional[np.ndarray]:
    """Read origin from the dataset's parent group, falling back to root.
    This is used ONLY to warn if the file's origin differs from the standard formula."""
    origin = None
    if "/" in dataset_path:
        parent = dataset_path.rsplit("/", 1)[0]
        if parent in f:
            origin = f[parent].attrs.get("origin", None)
    if origin is None:
        origin = f.attrs.get("origin", None)
    if origin is not None:
        origin = np.asarray(origin, dtype=np.float64)
    return origin


def _list_all_datasets(f: h5py.File) -> List[str]:
    """List all datasets in an h5py file."""
    paths = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    f.visititems(_visitor)
    return paths


def load_cmap(
    path: str,
    dataset_path: Optional[Union[str, List[str]]],
    expected_shape: Tuple[int, int, int],
    num_channels: int = 5,
    fallback_step: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a multi-channel 3D grid from a .cmap (h5py) file.

    The origin is ALWAYS computed from the standard-frame formula:
        origin = -(expected_shape * step) / 2
    This ensures alignment with PDB coordinates regardless of per-file cropping.

    Returns:
        grid:   (C, D, H, W) float32 where C == num_channels
        origin: (3,) spatial origin of the returned grid
        step:   (3,) voxel spacing
    """
    with h5py.File(path, "r") as f:
        # ── Resolve what to load ──
        paths_to_load: List[str] = []
        reference_path: str = ""
        single_4d = False

        if dataset_path is None:
            available = _list_all_datasets(f)
            candidates_3d = [p for p in available if f[p].ndim == 3]
            candidates_3d.sort()

            if not candidates_3d:
                raise KeyError(
                    f"No 3D datasets found in {path}.\n"
                    f"Available datasets: {available}"
                )

            if len(candidates_3d) != num_channels:
                print(
                    f"  ⚠ {path}: found {len(candidates_3d)} 3D datasets, "
                    f"expected {num_channels}"
                )
                if len(candidates_3d) > num_channels:
                    print(f"      Using first {num_channels}: {candidates_3d[:num_channels]}")
                    paths_to_load = candidates_3d[:num_channels]
                else:
                    raise ValueError(
                        f"Only {len(candidates_3d)} 3D datasets found, "
                        f"need {num_channels}"
                    )
            else:
                paths_to_load = candidates_3d

            reference_path = paths_to_load[0]

        elif isinstance(dataset_path, str):
            dset = f[dataset_path]
            if dset.ndim == 4:
                # Single 4D dataset — verify channel dimension
                if dset.shape[0] == num_channels:
                    single_4d = True
                    reference_path = dataset_path
                elif dset.shape[-1] == num_channels:
                    single_4d = True
                    reference_path = dataset_path
                else:
                    raise ValueError(
                        f"4D dataset shape {dset.shape} has no dimension "
                        f"equal to num_channels={num_channels}"
                    )
            elif dset.ndim == 3:
                # Single 3D dataset — not enough for multi-channel
                raise ValueError(
                    f"Single 3D dataset provided but num_channels={num_channels}. "
                    f"Pass a list of {num_channels} paths or set num_channels=1."
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_path} has ndim={dset.ndim}, expected 3 or 4"
                )

        else:
            # Explicit list of paths
            paths_to_load = list(dataset_path)
            if len(paths_to_load) != num_channels:
                raise ValueError(
                    f"Provided {len(paths_to_load)} paths but num_channels={num_channels}"
                )
            for dp in paths_to_load:
                if f[dp].ndim != 3:
                    raise ValueError(
                        f"Dataset {dp} has ndim={f[dp].ndim}, expected 3"
                    )
            reference_path = paths_to_load[0]

        # ── Read step from file ──
        step = _read_step_attr(f, reference_path)
        if step is None:
            print(f"  ⚠ {path}: 'step' not found in CMAP metadata — using fallback {fallback_step}")
            step = np.full(3, fallback_step, dtype=np.float64)

        # ── Compute standard-frame origin from formula ──
        # origin = -(size * step) / 2
        origin = -(np.array(expected_shape, dtype=np.float64) * step) / 2.0

        # Warn if file origin differs (indicates non-standard cropping)
        file_origin = _read_file_origin_attr(f, reference_path)
        if file_origin is not None:
            diff = np.linalg.norm(file_origin - origin)
            if diff > 0.1:
                print(f"  ℹ {path}: file origin {file_origin} differs from standard {origin}")
                print(f"     Using formula: -(size*step)/2")

        # ── Load data ──
        if single_4d:
            dset = f[dataset_path]
            if dset.shape[0] == num_channels:
                grid = dset[()].astype(np.float32)  # (C, D, H, W)
            else:
                # (D, H, W, C) → (C, D, H, W)
                grid = np.moveaxis(dset[()].astype(np.float32), -1, 0)
        else:
            grids = []
            for dp in paths_to_load:
                grids.append(f[dp][()].astype(np.float32))
            grid = np.stack(grids, axis=0)  # (C, D, H, W)

    grid = grid.astype(np.float32)

    # Guard against NaN / Inf
    for c in range(grid.shape[0]):
        ch = grid[c]
        nan_count = np.isnan(ch).sum()
        inf_count = np.isinf(ch).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠ {path} ch{c}: {nan_count} NaN, {inf_count} Inf — replacing with 0")
            grid[c] = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad or crop to uniform shape
    original_shape = grid.shape[-3:]
    if original_shape != expected_shape:
        grid, shifts = centre_pad_or_crop(grid, expected_shape, pad_value=0.0)

        is_crop = any(s > t for s, t in zip(original_shape, expected_shape))
        is_pad = any(s < t for s, t in zip(original_shape, expected_shape))
        action = (
            "cropped" if is_crop and not is_pad else
            "padded" if is_pad and not is_crop else
            "padded+cropped"
        )
        #print(f"  ℹ {path}: spatial {original_shape} → {expected_shape} (centre {action})")
        #print(f"     Standard origin: {origin}  (formula, not adjusted by shifts)")

        # Warn about aggressive cropping
        crop_dims = [s - t for s, t in zip(original_shape, expected_shape) if s > t]
        if crop_dims:
            max_crop = max(crop_dims)
            if max_crop > 16:
                print(f"  ⚠ {path}: cropping {max_crop} voxels from at least one dimension — "
                      f"pocket data may be lost. Consider increasing expected_grid_shape.")

    # Sparsity warning
    #for c in range(grid.shape[0]):
    #    zero_frac = (grid[c] == 0).mean()
    #    if zero_frac > 0.95:
    #        print(f"  ⚠ {path} ch{c}: {zero_frac * 100:.1f}% zeros — file may be corrupt or empty")

    return grid, origin, step


def normalise_mif(grid: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """
    Normalise a multi-channel 3D grid.

    Args:
        grid: (C, D, H, W) or (D, H, W) float32
        cfg:  DataConfig with normalisation settings

    Returns:
        Same shape, normalised (per-channel if multi-channel)
    """
    if grid.ndim == 3:
        grid = grid[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    out = np.empty_like(grid)

    for c in range(grid.shape[0]):
        ch = grid[c]

        if cfg.mif_norm == "global":
            ch = ch / cfg.mif_global_max
            ch = np.clip(ch, -1.0, 1.0)

        elif cfg.mif_norm == "minmax":
            lo, hi = ch.min(), ch.max()
            if hi - lo > 1e-8:
                ch = (ch - lo) / (hi - lo)
            else:
                ch[:] = 0.0

        elif cfg.mif_norm == "standard":
            mu, sigma = ch.mean(), ch.std()
            if sigma > 1e-8:
                ch = (ch - mu) / sigma
            else:
                ch[:] = 0.0

        elif cfg.mif_norm == "log1p":
            # log1p requires non-negative input — shift first
            if ch.min() < 0:
                ch = ch - ch.min()
            ch = np.log1p(ch)
            lo, hi = ch.min(), ch.max()
            if hi - lo > 1e-8:
                ch = (ch - lo) / (hi - lo)
            else:
                ch[:] = 0.0

        elif cfg.mif_norm == "none":
            pass

        else:
            raise ValueError(f"Unknown mif_norm: {cfg.mif_norm}")

        out[c] = ch

    return out[0] if squeeze else out


# ============================================================
# PDB loading and normalisation
# ============================================================

# Two-letter elements that can appear in organic ligand PDBs.
# Deliberately excludes Ca, Na, Fe, Zn, Mg, Mn — these are metals
# that could match residue labels (e.g. "CA" = alpha-carbon, not Calcium).
_LIGAND_TWO_LETTER_ELEMENTS = ["Cl", "Br", "Si", "Se"]


def _parse_element(line: str) -> str:
    """
    Extract element symbol from a PDB ATOM/HETATM line.

    Tries the element field (columns 76-78) first, then falls back
    to inferring from the atom name (columns 12-16).
    """
    # Try element field (PDB spec: columns 77-78, 1-indexed → 76:78, 0-indexed)
    if len(line) >= 78:
        elem = line[76:78].strip()
        if elem and elem[0].isalpha():
            # Normalise case: "CL" → "Cl", "BR" → "Br"
            if len(elem) == 2:
                return elem[0].upper() + elem[1].lower()
            return elem.capitalize()

    # Fallback: infer from atom name (columns 12-16, 0-indexed)
    atom_name = line[12:16].strip()
    if not atom_name:
        return "C"

    # Check two-letter elements first (e.g. "CL" in atom name → "Cl")
    for two in _LIGAND_TWO_LETTER_ELEMENTS:
        if atom_name.startswith(two) or atom_name.startswith(two.upper()):
            return two

    # Default: first alphabetic character
    for ch in atom_name:
        if ch.isalpha():
            return ch.capitalize()
    return "C"


def load_pdb(
    path: str,
    max_atoms: int,
    elem_to_id: Dict[str, int],
    unk_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Parse a ligand PDB file → padded atom coords, type IDs, mask, and raw elements.

    Returns:
        coords:    (max_atoms, 3) float32  — zero-padded
        type_ids:  (max_atoms,)   long     — 0 = pad, unk_id = unknown element
        mask:      (max_atoms,)   bool     — True for real atoms
        elements:  list of str             — raw element strings (for diagnostics)
    """
    coords_list = []
    types_list = []
    elements_list = []

    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue  # skip malformed coordinate lines

                elem = _parse_element(line)
                coords_list.append([x, y, z])
                types_list.append(elem_to_id.get(elem, unk_id))
                elements_list.append(elem)

    n_atoms = len(coords_list)

    # Initialise padded arrays
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    type_ids = np.zeros(max_atoms, dtype=np.int64)  # 0 = pad
    mask = np.zeros(max_atoms, dtype=bool)

    if n_atoms == 0:
        print(f"  ⚠ {path}: no ATOM/HETATM records found")
        return coords, type_ids, mask, []

    if n_atoms > max_atoms:
        print(f"  ⚠ {path}: {n_atoms} atoms > max_atoms={max_atoms}, truncating")
        n_atoms = max_atoms

    coords[:n_atoms] = coords_list[:n_atoms]
    type_ids[:n_atoms] = types_list[:n_atoms]
    mask[:n_atoms] = True

    # Report unknown elements
    unk_count = sum(1 for e in elements_list[:n_atoms] if e not in elem_to_id)
    if unk_count > 0:
        unk_elems = set(e for e in elements_list[:n_atoms] if e not in elem_to_id)
        print(f"  ⚠ {path}: {unk_count} atoms with unknown elements: {unk_elems}")

    return coords, type_ids, mask, elements_list[:n_atoms]


def normalise_coords(
    coords: np.ndarray,
    mask: np.ndarray,
    cfg: DataConfig,
    origin: Optional[np.ndarray] = None,
    step: Optional[np.ndarray] = None,
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Normalise atom coordinates to align with the MIF grid.

    Args:
        coords:     (max_atoms, 3) float32 — raw Å coordinates
        mask:       (max_atoms,) bool
        cfg:        DataConfig
        origin:     (3,) spatial origin of the grid (from standard formula). If None, uses cfg fallback.
        step:       (3,) voxel spacing (from CMAP file). If None, uses cfg fallback.
        grid_shape: (3,) actual shape of the returned grid. If None, uses cfg.expected_grid_shape.

    Returns:
        (max_atoms, 3) float32 — normalised (padding atoms stay 0)
    """
    if origin is None:
        origin = np.array(cfg.grid_origin, dtype=np.float64)
    if step is None:
        step = np.full(3, cfg.voxel_spacing, dtype=np.float64)
    if grid_shape is None:
        grid_shape = cfg.expected_grid_shape

    result = coords.copy()
    real = mask.nonzero()[0]
    if len(real) == 0:
        return result

    real_coords = coords[real]

    if cfg.coord_norm == "centre_scale":
        # Compute physical extent from actual grid metadata
        physical_extent = step * (np.array(grid_shape, dtype=np.float64) - 1)
        centre = origin + physical_extent / 2.0
        half_extent = physical_extent / 2.0

        result[real] = (real_coords - centre) / half_extent

    elif cfg.coord_norm == "voxel_index":
        result[real] = (real_coords - origin) / step

    elif cfg.coord_norm == "none":
        pass  # keep raw Å

    else:
        raise ValueError(f"Unknown coord_norm: {cfg.coord_norm}")

    return result


# ============================================================
# Dataset
# ============================================================

class MIFLigandDataset(Dataset):
    """
    Paired dataset: multi-channel MIF grid ↔ ligand PDB.

    Expects sample directories like:
        dataset_dir/ligand1/ligand1.cmap
        dataset_dir/ligand1/ligand1.pdb
    """

    def __init__(self, sample_dirs: List[str], cfg: DataConfig):
        self.sample_dirs = sample_dirs
        self.cfg = cfg
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cfg.cache and idx in self._cache:
            return self._cache[idx]

        #sample_dir = self.sample_dirs[idx]
        #stem = os.path.basename(sample_dir)

        #cmap_path = os.path.join(sample_dir, f"{stem}.cmap")
        #pdb_path = os.path.join(sample_dir, f"{stem}.pdb")

        sample_dir = self.sample_dirs[idx]

        cmap_files = [f for f in os.listdir(sample_dir) if f.endswith(".cmap")]
        pdb_files = [f for f in os.listdir(sample_dir) if f.endswith(".pdb")]

        if len(cmap_files) != 1 or len(pdb_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .cmap and one .pdb in {sample_dir}, "
                f"found {len(cmap_files)} .cmap and {len(pdb_files)} .pdb"
            )

        cmap_path = os.path.join(sample_dir, cmap_files[0])
        pdb_path = os.path.join(sample_dir, pdb_files[0])

        if not os.path.isfile(cmap_path):
            raise FileNotFoundError(f"Missing .cmap for sample {stem}: {cmap_path}")
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"Missing .pdb for sample {stem}: {pdb_path}")

        # --- Load MIF ---
        grid, origin, step = load_cmap(
            cmap_path,
            self.cfg.dataset_path,
            self.cfg.expected_grid_shape,
            num_channels=self.cfg.num_channels,
            fallback_step=self.cfg.voxel_spacing,
        )
        grid = normalise_mif(grid, self.cfg)
        mif_tensor = torch.from_numpy(grid)  # (C, D, H, W)

        # --- Load PDB ---
        coords, type_ids, mask, _ = load_pdb(
            pdb_path,
            self.cfg.max_atoms,
            self.cfg.elem_to_id,
            self.cfg.unk_id,
        )
        coords = normalise_coords(
            coords, mask, self.cfg,
            origin=origin, step=step, grid_shape=grid.shape[-3:]
        )

        sample = {
            "mif_grid": mif_tensor,
            "atom_coords": torch.from_numpy(coords),
            "atom_types": torch.from_numpy(type_ids),
            "atom_mask": torch.from_numpy(mask),
            "origin": torch.from_numpy(origin),
            "step": torch.from_numpy(step),
        }

        if self.cfg.cache:
            self._cache[idx] = sample

        return sample

def discover_pairs(
    dataset_dir: str,
    cmap_ext: str = ".cmap",
    pdb_ext: str = ".pdb",
) -> List[str]:
    """
    Discover sample directories containing both .cmap and .pdb files.

    Expects:
        dataset_dir/
            mol_28/
                ligand28.cmap
                ligand28.pdb
            mol_31/
                ligand31.cmap
                ligand31.pdb

    Returns:
        Sorted list of sample directory paths.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    sample_dirs = []
    for entry in sorted(os.listdir(dataset_dir)):
        sample_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(sample_dir):
            continue

        cmap_files = [f for f in os.listdir(sample_dir) if f.endswith(cmap_ext)]
        pdb_files = [f for f in os.listdir(sample_dir) if f.endswith(pdb_ext)]

        if len(cmap_files) == 1 and len(pdb_files) == 1:
            sample_dirs.append(sample_dir)
        elif len(cmap_files) == 0 and len(pdb_files) == 0:
            pass  # silently skip empty dirs
        else:
            missing = []
            if len(cmap_files) == 0:
                missing.append(cmap_ext)
            elif len(cmap_files) > 1:
                missing.append(f"multiple *{cmap_ext}")
            if len(pdb_files) == 0:
                missing.append(pdb_ext)
            elif len(pdb_files) > 1:
                missing.append(f"multiple *{pdb_ext}")
            print(f"  ⚠ Skipping {entry}: {', '.join(missing)}")

    if not sample_dirs:
        raise FileNotFoundError(
            f"No valid sample directories found in {dataset_dir}. "
            f"Expected subdirs each containing one {cmap_ext} and one {pdb_ext} file."
        )

    print(f"Found {len(sample_dirs)} matched pairs in {dataset_dir}.")
    return sample_dirs

def _worker_init_fn(worker_id: int):
    
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_dataloaders(
    cfg: Optional[DataConfig] = None,
    dataset_dir: Optional[str] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train and val DataLoaders from a per-sample directory structure.

    Args:
        dataset_dir: Root directory containing sample subdirectories.
                     If None, uses cfg.dataset_dir.
        cfg: DataConfig instance.

    Returns:
        (train_loader, val_loader) or (None, None) if data cannot be loaded.
    """
    if cfg is None:
        cfg = DataConfig()

    dataset_dir = dataset_dir or cfg.dataset_dir
    if not dataset_dir:
        print("\n❌ dataset_dir not configured.")
        print("   Please set DataConfig.dataset_dir or pass dataset_dir to build_dataloaders().\n")
        return None, None

    if not os.path.isdir(dataset_dir):
        print(f"\n❌ dataset_dir does not exist: {dataset_dir}")
        return None, None

    try:
        sample_dirs = discover_pairs(dataset_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return None, None

    n = len(sample_dirs)
    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_fraction))

    if n_val >= n:
        print(f"⚠️  Dataset has only {n} sample(s) — using it for both train and val.")
        train_idx = indices
        val_idx = indices
    else:
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    train_dataset = MIFLigandDataset([sample_dirs[i] for i in train_idx], cfg)
    val_dataset = MIFLigandDataset([sample_dirs[i] for i in val_idx], cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )

    print(cfg.summary())
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    return train_loader, val_loader


# ============================================================
# CLI: quick test for a single pair
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data.py on a single .cmap / .pdb pair")
    parser.add_argument("--cmap", required=True, help="Path to .cmap file")
    parser.add_argument("--pdb",  required=True, help="Path to .pdb file")
    parser.add_argument("--mif-norm", default=None,
                        help="Override MIF normalisation (global/minmax/standard/log1p/none)")
    parser.add_argument("--coord-norm", default=None,
                        help="Override coord normalisation (centre_scale/voxel_index/none)")
    parser.add_argument("--model-config", action="store_true",
                        help="Test DataConfig → CVaeConfig round-trip and print model config")
    args = parser.parse_args()

    cfg = DataConfig()
    if args.mif_norm:
        cfg.mif_norm = args.mif_norm
    if args.coord_norm:
        cfg.coord_norm = args.coord_norm

    print("=" * 60)
    print(cfg.summary())

    # ── Test DataConfig → CVaeConfig round-trip ──
    if args.model_config:
        print()
        print("─" * 60)
        print("DataConfig → CVaeConfig round-trip:")
        try:
            model_cfg = cfg.to_model_config()
            print(f"  input_shape:     {model_cfg.input_shape}")
            print(f"  in_channels:     {model_cfg.in_channels}")
            print(f"  num_atom_types:  {model_cfg.num_atom_types}  "
                  f"(data: {cfg.num_atom_types})")
            print(f"  max_atoms:       {model_cfg.max_atoms}  "
                  f"(data: {cfg.max_atoms})")
            print(f"  padding_idx:     {model_cfg.padding_idx}  "
                  f"(data pad_id: {cfg.pad_id})")

            # Verify all data-dependent fields match
            assert model_cfg.input_shape == cfg.expected_grid_shape
            assert model_cfg.in_channels == cfg.num_channels
            assert model_cfg.num_atom_types == cfg.num_atom_types
            assert model_cfg.max_atoms == cfg.max_atoms
            assert model_cfg.padding_idx == cfg.pad_id

            # Test sub-config derivation
            from model import _build_ligand_config
            lig_cfg = _build_ligand_config(model_cfg)
            print(f"\n  LigandEncoderConfig (derived):")
            print(f"    num_atom_types: {lig_cfg.num_atom_types}  "
                  f"(data: {cfg.num_atom_types})")
            print(f"    max_atoms:      {lig_cfg.max_atoms}  "
                  f"(data: {cfg.max_atoms})")
            print(f"    padding_idx:    {lig_cfg.padding_idx}  "
                  f"(data pad_id: {cfg.pad_id})")
            assert lig_cfg.num_atom_types == cfg.num_atom_types
            assert lig_cfg.max_atoms == cfg.max_atoms
            assert lig_cfg.padding_idx == cfg.pad_id

            print("\n  ✓ All data-dependent fields match.")
        except Exception as e:
            print(f"\n  ✗ Round-trip failed: {e}")

    # ── Test cmap loading with pad/crop ──
    print()
    print("Loading .cmap ...")
    grid, origin, step = load_cmap(
        args.cmap, cfg.dataset_path, cfg.expected_grid_shape,
        num_channels=cfg.num_channels,
        fallback_step=cfg.voxel_spacing,
    )
    print(f"  Output shape:  {grid.shape}  (target: {cfg.expected_grid_shape})")
    print(f"  Origin:        {origin}")
    print(f"  Step:          {step}")
    print(f"  Range:         [{grid.min():.4f}, {grid.max():.4f}]")
    print(f"  Mean:          {grid.mean():.4f}")
    print(f"  Sparsity:      {(grid == 0).mean() * 100:.1f}% zeros")

    print()
    grid_norm = normalise_mif(grid.copy(), cfg)
    print(f"  Norm strategy: {cfg.mif_norm}")
    print(f"  Norm range:    [{grid_norm.min():.4f}, {grid_norm.max():.4f}]")
    print(f"  Norm mean:     {grid_norm.mean():.4f}")

    # ── Test PDB loading ──
    print()
    print("Loading .pdb ...")
    coords, type_ids, mask, elements = load_pdb(
        args.pdb, cfg.max_atoms, cfg.elem_to_id, cfg.unk_id
    )
    n_real = mask.sum()
    print(f"  Atoms:        {n_real} / max {cfg.max_atoms}")
    if n_real > 0:
        from collections import Counter
        elem_counts = Counter(elements)
        print(f"  Elements:     {dict(elem_counts)}")

        # Check for unknowns
        unk_mask = type_ids[:n_real] == cfg.unk_id
        if unk_mask.any():
            unk_elems = [elements[i] for i in range(n_real) if type_ids[i] == cfg.unk_id]
            print(f"  ⚠ Unknown elements mapped to unk_id={cfg.unk_id}: {set(unk_elems)}")

        print(f"  Raw coord range (real atoms):")
        real_c = coords[mask]
        for ax, name in enumerate("XYZ"):
            print(f"    {name}: [{real_c[:, ax].min():.2f}, {real_c[:, ax].max():.2f}]")

    print()
    coords_norm = normalise_coords(coords, mask, cfg, origin=origin, step=step, grid_shape=grid.shape[-3:])
    if n_real > 0:
        real_n = coords_norm[mask]
        print(f"  Norm strategy: {cfg.coord_norm}")
        print(f"  Norm coord range (real atoms):")
        for ax, name in enumerate("XYZ"):
            print(f"    {name}: [{real_n[:, ax].min():.4f}, {real_n[:, ax].max():.4f}]")

    # ── Test dataset sample ──
    print()
    print("Dataset sample ...")
    dataset = MIFLigandDataset([args.cmap], [args.pdb], cfg)
    sample = dataset[0]
    for k, v in sample.items():
        print(f"  {k:12s}: shape={tuple(v.shape):20s} dtype={v.dtype}  "
              f"range=[{v.min():.4f}, {v.max():.4f}]")

    # Verify cache works
    if cfg.cache:
        sample2 = dataset[0]
        assert sample["mif_grid"].data_ptr() == sample2["mif_grid"].data_ptr(), \
            "Cache not working — samples are different objects"
        print("  Cache:        ✓ (second access returns same object)")

    print()
    print("=" * 60)
    print("✓ All checks passed. Update § parameters as needed.")
