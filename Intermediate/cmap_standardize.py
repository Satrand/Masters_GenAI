#!/usr/bin/env python3

import argparse
import os
import h5py
import numpy as np

# Collect all 3D grids in a file 
def get_grids(fin):
    grids = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            grids.append(obj)

    fin.visititems(visit)
    return grids

# Compute global shape across all files
def compute_global_shape(file_list):
    shapes = []

    for fp in file_list:
        with h5py.File(fp, "r") as f:
            grids = get_grids(f)
            if grids:
                shapes.append(np.max([g.shape for g in grids], axis=0))

    return tuple(np.max(shapes, axis=0))

# Center padding
def pad_to_shape(arr, target_shape):
    out = np.zeros(target_shape, dtype=arr.dtype)

    offsets = [(t - s) // 2 for s, t in zip(arr.shape, target_shape)]

    slices = tuple(
        slice(o, o + s) for o, s in zip(offsets, arr.shape)
    )

    out[slices] = arr
    return out

# Pipeline
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="folder containing .cmap files OR single file")
    parser.add_argument("output", help="output folder")
    parser.add_argument("--global_shape", nargs=3, type=int, default=None)

    args = parser.parse_args()

    #gather input files
    if os.path.isdir(args.input):
        file_list = sorted([
            os.path.join(r, f)
            for r, _, fs in os.walk(args.input)
            for f in fs if f.endswith(".cmap")
        ])
    else:
        file_list = [args.input]

    os.makedirs(args.output, exist_ok=True)

    #determine global grid shape
    if args.global_shape:
        target_shape = tuple(args.global_shape)
        print("Using manual global shape:", target_shape)
    else:
        target_shape = compute_global_shape(file_list)
        print("Computed global shape:", target_shape)

    # Process files
    for file_idx, in_file in enumerate(file_list):

        with h5py.File(in_file, "r") as f:
            grids = get_grids(f)

            if len(grids) == 0:
                print(f"Skipping {in_file}: no grids found")
                continue

            print(f"\n[{file_idx+1}/{len(file_list)}] Processing: {in_file}")
            print(f"Number of channels (grids): {len(grids)}")

            # SINGLE CHANNEL SUPERPOSITION OUTPUT
            out = np.zeros(target_shape, dtype=np.float64)

            for i, g in enumerate(grids):
                arr = g[:]

                # padding to global shape
                if arr.shape != target_shape:
                    arr = pad_to_shape(arr, target_shape)

                # SUM INTO SINGLE CHANNEL
                out += arr

                if (i + 1) % 5 == 0 or i == len(grids) - 1:
                    print(f"  processed {i+1}/{len(grids)}")

        # write per-molecule output 
        base = os.path.basename(in_file).replace(".cmap", "")
        out_file = os.path.join(args.output, f"{base}_superposition.cmap")

        with h5py.File(out_file, "w") as f:
            grp = f.create_group("Chimera")

            ds = grp.create_dataset(
                "smif_superposition",
                data=out.astype(np.float32),
                compression="gzip"
            )

            ds.attrs["n_channels"] = len(grids)
            ds.attrs["grid_shape"] = np.array(target_shape)
            ds.attrs["mode"] = "single_channel_superposition"
        
        print("Wrote:", out_file)

if __name__ == "__main__":
    main()

