import numpy as np
import os
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import AtomPDBResidueInfo
from rdkit.Chem import AllChem, Crippen, GetSymmSSSR, rdMolDescriptors

## 3D Embedding and Optimization
def smiles_to_3d(smiles, optimize=True, forcefield="MMFF", random_seed=42): #optimize:bool >
    #Parse SMILES (i.e. read SMILES text and convert into an internal molecular graph repre>
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    #Add hydrogens
    mol = Chem.AddHs(mol)

    #Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    res = AllChem.EmbedMolecule(mol, params)

    if res != 0:
        raise RuntimeError("3D embedding failed")

    # Optimize geometry
    if optimize:
        if forcefield.upper() == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

    return mol

## Center Molecule
def center_molecule(mol):
    conf = mol.GetConformer()
    com = np.zeros(3)
    total_mass = 0
    for atom in mol.GetAtoms():
        mass = atom.GetMass() if atom.GetMass() > 0 else 1.0
        pos = conf.GetAtomPosition(atom.GetIdx())
        com += mass * np.array([pos.x, pos.y, pos.z])
        total_mass += mass
    com /= total_mass

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        conf.SetAtomPosition(atom.GetIdx(), tuple(np.array([pos.x, pos.y, pos.z]) - com))

    return mol


## Bounding Box
def get_bounding_box(mol):
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    return min_coords, max_coords

## Align to Principal Axes
def align_to_principal_axes(mol):
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    inertia = np.zeros((3,3))
    for i, atom in enumerate(mol.GetAtoms()):
        mass = atom.GetMass() if atom.GetMass() > 0 else 1.0
        r = coords[i]

        inertia[0,0] += mass*(r[1]**2 + r[2]**2)
        inertia[1,1] += mass*(r[0]**2 + r[2]**2)
        inertia[2,2] += mass*(r[0]**2 + r[1]**2)
        inertia[0,1] -= mass*r[0]*r[1]
        inertia[0,2] -= mass*r[0]*r[2]
        inertia[1,2] -= mass*r[1]*r[2]

    inertia[1,0] = inertia[0,1]
    inertia[2,0] = inertia[0,2]
    inertia[2,1] = inertia[1,2]

    eigvals, eigvecs = np.linalg.eigh(inertia)
    rotated_coords = np.dot(coords, eigvecs)

    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(rotated_coords[i]))

    return mol


## Atom Names
def generate_atom_names(mol):
    names = []
    for atom in mol.GetAtoms():
        names.append(f"{atom.GetSymbol()}{atom.GetIdx() + 1}")
    return names


## Crippen Hydrophobicity
def atom_hphobicity(mol):
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    return [round(c[0],4) for c in contribs]
## Partial Charges
def partial_charges_from_mol(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        try:
            q = atom.GetDoubleProp('_GasteigerCharge')
        except:
            q = 0.0
        charges.append(round(q,4))
    return charges


## Aromatic Rings
def get_aromatic_rings(mol):
    rings = GetSymmSSSR(mol)
    aromatic_rings = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings.append(list(ring))
    return aromatic_rings

## HBA/HBD (unchanged)
def detect_acceptors(mol):
    acceptors = []
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        if Z not in (7, 8):
            continue
        if atom.GetFormalCharge() > 0:
            continue
        if atom.GetTotalDegree() == 4 and Z == 7:
            continue
        acceptors.append(atom.GetIdx())
    return sorted(set(acceptors))


def detect_donors(mol):
    donors = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (7,8):
            if any(n.GetAtomicNum()==1 for n in atom.GetNeighbors()):
                donors.append(atom.GetIdx())
    return sorted(donors)


def is_sp2(atom):
    return atom.GetHybridization() == Chem.HybridizationType.SP2

def construct_hbond_string(mol, atom_idx, is_donor):
    atom = mol.GetAtomWithIdx(atom_idx)
    atom_name = atom.GetSymbol() + str(atom_idx + 1)
    heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum()>1]

    if is_donor:
        if len(heavy_neighbors) == 0:
            return atom_name
        tails = ".".join([n.GetSymbol() + str(n.GetIdx()+1) for n in heavy_neighbors])
        hbond_str = f"{atom_name}={tails}->{atom_name}"
    else:
        if heavy_neighbors:
            tail = heavy_neighbors[0]
            hbond_str = f"{atom_name}={tail.GetSymbol()}{tail.GetIdx()+1}->{atom_name}"
        else:
            hbond_str = atom_name

    if is_sp2(atom):
        hbond_str += "!"

    return hbond_str



def set_pdb_info(mol, resname="LIG", chainid="A", resid=1):
    names = generate_atom_names(mol)
    for idx, atom in enumerate(mol.GetAtoms()):
        info = AtomPDBResidueInfo()
        info.SetName(names[idx])
        info.SetResidueName(resname if atom.GetAtomicNum()>1 else "UNL")
        info.SetResidueNumber(resid)
        info.SetChainId(chainid)
        atom.SetMonomerInfo(info)
    return mol


def write_pdb_file(mol, filename="ligand.pdb"):
    pdb_block = Chem.MolToPDBBlock(mol)
    with open(filename, "w") as f:
        f.write(pdb_block)

def write_table_file(mol, resname="LIG", filename="table.chem"):
    mol = center_molecule(mol)
    mol = align_to_principal_axes(mol)

    names = generate_atom_names(mol)
    hydrophob = atom_hphobicity(mol)
    rings = get_aromatic_rings(mol)
    acceptors = detect_acceptors(mol)
    donors = detect_donors(mol)
    min_box, max_box = get_bounding_box(mol)

    outstr = f"[SELECTION_QUERY]\nresname {resname}\n\n"
    outstr += "[ATOM_HPHOBICITY]\n"

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1: #skip hydrogens
            continue
        idx = atom.GetIdx()
        outstr += f"{resname}/{names[idx]}: {hydrophob[idx]:.4f}\n"

    outstr += "\n[NAMES_STACKING]\n"
    for ring in rings:
        ring_names = [names[i] for i in ring]
        outstr += f"{resname}: " + " ".join(ring_names) + "\n"

    outstr += "\n[NAMES_HBACCEPTORS]\n"
    acceptor_strings = [construct_hbond_string(mol, idx, False) for idx in acceptors]
    outstr += f"{resname}: " + " ".join(acceptor_strings) + "\n"
#    outstr += f"{resname}: " + " ".join(map(str, acceptors)) + "\n"

    outstr += "\n[NAMES_HBDONORS]\n"
    donor_strings = [construct_hbond_string(mol, idx, True) for idx in donors]
    outstr += f"{resname}: " + " ".join(donor_strings) + "\n"
#    outstr += f"{resname}: " + " ".join(map(str, donors)) + "\n"

    with open(filename, "w") as f:
        f.write(outstr)


# Pipeline

input_csv = "filtered_organic_smiles.csv"
output_dir = "output"
n_mols = 2000
start = 0

df = pd.read_csv(input_csv)
# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs(output_dir, exist_ok=True)

records = []
failed = []

t0 = time.time()

success_count = 0
i = 0

while success_count < n_mols and i < len(df):

    smiles = df.iloc[i][ "canonical_smiles"]
    mol_id = start + success_count + 1

    mol_dir = os.path.join(output_dir, f"mol_{mol_id}")

    try:
        mol = smiles_to_3d(smiles)
        mol = set_pdb_info(mol)

        os.makedirs(mol_dir, exist_ok=True)

        pdb_file = os.path.join(mol_dir, f"ligand{mol_id}.pdb")
        chem_file = os.path.join(mol_dir, f"table{mol_id}.chem")

        write_pdb_file(mol, pdb_file)
        write_table_file(mol, filename=chem_file)

        records.append({"mol_id": mol_id, "smiles": smiles})
        success_count += 1

    except Exception as e:
        print(f"Skipping molecule {mol_id}: {e}")
        failed.append({"mol_id": mol_id, "smiles": smiles, "error": str(e)})

    i += 1

t1 = time.time()

pd.DataFrame(records).to_csv(os.path.join(output_dir, "molecule_list.csv"), index=False)
pd.DataFrame(failed).to_csv(os.path.join(output_dir, "failed_molecules.csv"), index=False)

print("Success:", len(records))
print("Failed:", len(failed))
print("Time:", t1 - t0)
