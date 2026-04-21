import pandas as pd
from rdkit import Chem
import pandas as pd

input_path = "/scratch/prj/rcmb_genai_transition/chembl_36/chembl_36_sqlite/chembl_36_smile>
output_path = "filtered_organic_smiles.csv"

excluded_atoms = {5, 14, 33, 34}  # B, Si, As, Se

def filter_molecule(smiles, max_heavy_atoms=34):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    atoms = list(mol.GetAtoms())

    # must contain carbon
    if not any(a.GetAtomicNum() == 6 for a in atoms):
        return False

    # exclude unwanted elements
    if any(a.GetAtomicNum() in excluded_atoms for a in atoms):
        return False

    # size filter
    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return False

    return True


print("Loading dataset...")
df = pd.read_csv(input_path)

print("Removing multi-molecules...")
df = df[~df['canonical_smiles'].str.contains(r'\.', regex=True)].copy()

print("Filtering molecules...")
df = df[df['canonical_smiles'].apply(filter_molecule)].copy()

print("Saving filtered dataset...")
df.to_csv(output_path, index=False)

print(f"Done! Remaining molecules: {len(df)}")
