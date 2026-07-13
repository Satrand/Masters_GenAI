#!/usr/bin/env python3

import argparse
from pathlib import Path
import h5py
import numpy as np

def inspect_file(fp):
    with h5py.File(fp, "r") as f:
        print("\n", "="*70)
        print("FILE:", fp)
        print("="*70)

        datasets = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                datasets.append(obj)

        f.visititems(visit)

        print("Channels:", len(datasets))

        shapes = set()

        for d in datasets:
            shapes.add(d.shape)
            data = d[:]
            print(f"{d.name:40} {d.shape} min={data.min():.3f} max={data.max():.3f}")

        print("Unique shapes:", shapes)

        if len(datasets) > 0:
            superpos = sum(d[:].astype(float) for d in datasets)
            print("\nSuperposition stats:")
            print("shape:", superpos.shape)
            print("min:", superpos.min(), "max:", superpos.max())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)

    if args.scan:
        files = list(path.rglob("*.cmap"))
    else:
        files = [path]

    for f in files:
        inspect_file(f)

if __name__ == "__main__":
    main()
