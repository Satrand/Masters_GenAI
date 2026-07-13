import numpy as np
import os
import time
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import AtomPDBResidueInfo
from rdkit.Chem import AllChem, Crippen, GetSymmSSSR, rdMolDescriptors

## 3D Embedding and Optimization
def smiles_to_3d(smiles, optimize=True, forcefield="MMFF", random_seed=42): #optimize:bo>
    #Parse SMILES (i.e. read SMILES text and convert into an internal molecular graph re>
    
    #print("!!!", smiles)
    resid = 1
    chainid = "A"
    resname="LIG"

    mol = Chem.MolFromSmiles(smiles)
    mol = adjust_pdb_atom_names(mol,resname,resid,chainid)
    atom = mol.GetAtomWithIdx(0)
    print(atom)
    print(atom.GetPropsAsDict())
    atom.GetProp("atom_name")


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

def adjust_pdb_atom_names(mol, resname, resid, chainid):
    #set residue name
    for idx,atom in enumerate(mol.GetAtoms()):
        residue_info = AtomPDBResidueInfo()
        residue_info.SetResidueName(resname)
        residue_info.SetChainId(chainid)
        residue_info.SetResidueNumber(resid)
        atom.SetAtomMapNum(idx + 1)  # Required for PDB residue info
        atom.SetPDBResidueInfo(residue_info)
    pdb_block = Chem.MolToPDBBlock(mol)
    atom_names = list()
    # get atom names from pdb block
    for line in pdb_block.split('\n'):
        if line.startswith('HETATM') or line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            atom_names.append(atom_name)
    # apply names to mol
    for idx,atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atom_name", atom_names[idx])
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


def get_stacking_string(resname, aromatic_rings, pi_systems):
    outstr = str()
    start = resname.strip() + ": "
    for ring in aromatic_rings:
        this_ring = str()
        for name in ring:
            this_ring += "{} ".format(name)
        outstr += start + this_ring + "\n"
    for system in pi_systems:
        for atomset in system:
            this_set = str()
            for name in atomset:
                this_set += "{} ".format(name)
            outstr += start + this_set + "\n"
    return outstr

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
    # get all rings in molecule
    rings = GetSymmSSSR(mol)
    aromatic_rings = [] 
    # check all the rings to see whether they are aromatic
    for ring in rings:
        is_aromatic_ring = all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx>
        if is_aromatic_ring:
            aromatic_rings.append([mol.GetAtomWithIdx(idx).GetProp("atom_name") for idx >
    return aromatic_rings

                                  ## HBA/HBD (unchanged)
def detect_acceptors(mol,complexT=True):
    #acceptors = []
    #for atom in mol.GetAtoms():
    #    Z = atom.GetAtomicNum()
    #    if Z not in (7, 8):
    #        continue
    #    if atom.GetFormalCharge() > 0:
    #        continue
    #    if atom.GetTotalDegree() == 4 and Z == 7:
    #        continue
    #    acceptors.append(atom.GetIdx())
    #return sorted(set(acceptors))
    if complexT:
        #hba_pattern = "[#8,#7,#16,#9,#17, #35,#53;H0,H1,H2]"
        hba_pattern = "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v>
    else:
        hba_pattern = "[#7,#8;!H0;!$(C#[#7,#8])]" 
        
    acceptors = mol.GetSubstructMatches(Chem.MolFromSmarts(hba_pattern))
    return list(set([atom_idx for acc in acceptors for atom_idx in acc]))

def detect_donors(mol):
    #nors = []
    #for atom in mol.GetAtoms():
    #    if atom.GetAtomicNum() in (7,8):
    #        if any(n.GetAtomicNum()==1 for n in atom.GetNeighbors()):
    #            donors.append(atom.GetIdx())
    #return sorted(donors)
    hbd_pattern = "[#8,#7,#16;H1,H2]"
    donors = mol.GetSubstructMatches(Chem.MolFromSmarts(hbd_pattern))
    return list(set([atom_idx for d in donors for atom_idx in d]))

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
        info.SetResidueName("LIG")
        info.SetResidueNumber(resid)
        info.SetChainId(chainid)
        atom.SetMonomerInfo(info)
    return mol

def find_pi_systems(mol):
    # Carboxylate pattern
    carboxylate_pattern = Chem.MolFromSmarts("[C](=[O])[O-]")
    carboxylate_matches = parse_matches(mol,mol.GetSubstructMatches(carboxylate_pattern))
    # Arginine side chain (guanidinium) pattern
    arginine_pattern = Chem.MolFromSmarts("[N;H0]([C])([N;H1,H2])[N;H1,H2]")
    arginine_matches = parse_matches(mol,mol.GetSubstructMatches(arginine_pattern))
    # Amidine pattern
    amidine_pattern = Chem.MolFromSmarts("[N;H0]([C])([N;H1,H2])")
    amidine_matches = parse_matches(mol,mol.GetSubstructMatches(amidine_pattern))
    # We are not detecting conjugated C-C chains here
    return carboxylate_matches, arginine_matches, amidine_matches


def write_pdb_file(mol, filename="ligand.pdb", resname="LIG"):
    mol = set_pdb_info(mol, resname=resname)

    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info:
            info.SetResidueName(resname)

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

    hydrophobicities = defaultdict(list)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1: #skip hydrogens
            continue
        idx = atom.GetIdx()
        hydrophobicities[hydrophob[idx]].append(names[idx])
    
    outstr += f"[HYDROPHOBICITY]\n{resname}:"
    for hphob_value, names in sorted(hydrophobicities.items(), key = lambda t: t[0]):
        outstr += f"{','.join(names)}={hphob_value:.4f} "
    outstr += "\n"

    # outstr += f"{names[idx]}={hydrophob[idx]:.4f}\n"

    outstr += f"\n[STACKING]\n{resname}: "
    for ring in rings:
        outstr += "-".join(ring) + " "
    outstr += "\n"

    outstr += "\n[HBACCEPTORS]\n"
    acceptor_strings = [construct_hbond_string(mol, idx, False) for idx in acceptors]
    outstr += f"{resname}: " + " ".join(acceptor_strings) + "\n"
#    outstr += f"{resname}: " + " ".join(map(str, acceptors)) + "\n"

    outstr += "\n[HBDONORS]\n"
    donor_strings = [construct_hbond_string(mol, idx, True) for idx in donors]
    outstr += f"{resname}: " + " ".join(donor_strings) + "\n"
#    outstr += f"{resname}: " + " ".join(map(str, donors)) + "\n"

    with open(filename, "w") as f:
        f.write(outstr)

# Pipeline for edge cases
edge_cases = {
    "pyrrole": "c1cc[nH]c1",
    "imidazole": "c1ncc[nH]1",
    "indole": "c1ccc2c(c1)[nH]cc2",
    "amide": "CC(=O)NC",
    "aniline": "c1ccccc1N",
    "pyridinium": "c1cc[nH+]cc1",
    "quinoline": "c1ccc2ncccc2c1",
    "purine": "c1ncnc2n(ncc2n1)",
    "histidine": "NCCc1ncc[nH]1",
    "pyridine": "c1ccncc1"
}

output_dir = "edge_cases_output"
os.makedirs(output_dir, exist_ok=True)

print("Running EDGE CASE pipeline only...\n")

for name, smi in edge_cases.items():
    print(f"Processing {name}: {smi}")

    # 1. Generate 3D structure using YOUR pipeline
    mol = smiles_to_3d(smi, optimize=True)

    # 2. Create folder per molecule
    mol_dir = os.path.join(output_dir, name)
    os.makedirs(mol_dir, exist_ok=True)

    # 3. Define output paths
    pdb_path = os.path.join(mol_dir, f"{name}.pdb")
    chem_path = os.path.join(mol_dir, f"{name}.chem")

    # 4. Write outputs using YOUR pipeline functions
    write_pdb_file(mol, pdb_path)
    write_table_file(mol, filename=chem_path)

print("\n=== DONE: all edge cases processed ===")

