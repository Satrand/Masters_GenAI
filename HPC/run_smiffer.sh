#!/bin/bash
#SBATCH --job-name=smiffer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=roeder_cpu
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate volgrids

for dir in output/mol_*
do
    pdb_file=("$dir"/*.pdb)
    chem_file=("$dir"/*.chem)

    pdb="${pdb_file[0]}"
    chem="${chem_file[0]}"

    echo "Running SMIFfer for $dir"
    echo "PDB:  $pdb"
    echo "CHEM: $chem"

    volgrids smiffer ligand "$pdb" \
        -b "$chem" \
        -o "$dir"
done

