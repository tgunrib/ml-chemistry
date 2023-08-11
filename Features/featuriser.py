import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator

# This code merges multiple dataset and cleans duplicates and null values
# All dataset used can keep their original names and are referenced in the project.


def extract_data(name, solubility_name):
    print(name)
    data = pd.read_csv(name)[['SMILES', solubility_name]]
    data.rename(columns={solubility_name: 'logS'}, inplace=True)
    print(data.head())
    return data


def merge_data(list_dataframe):
    data = pd.concat(list_dataframe, ignore_index=True).dropna()
    return data


def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles


def rdkit_descriptor(molecules):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    Mol_descriptors = []
    for mol in molecules:
        # add hydrogens to molecules
        mol = Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors, desc_names


def rdkit_fingerprints(molecules):
    Mol_descriptors = []
    for mol in molecules:
        # add hydrogens to molecules
        mol = Chem.AddHs(mol)
        generator = Chem.rdFingerprintGenerator.GetRDKitFPGenerator(5)
        # Calculate all 200 descriptors for each molecule
        fingerprints = [Chem.AllChem.GetAdjacencyMatrix(mol),
                        np.array(Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)),
                        np.array(Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)),
                        np.array(Chem.AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)),
                        generator.GetFingerprint(mol),
                        rdMolDescriptors.MQNs_(mol)
                        ]
        Mol_descriptors.append(fingerprints)
    return Mol_descriptors, ['adjacenyMatrix',
                             'atom_pair_fingerprint',
                             'morgan_fingerprint',
                             'topological_fingerpint',
                             'rdKitFingerPrints',
                             'MQNs']
