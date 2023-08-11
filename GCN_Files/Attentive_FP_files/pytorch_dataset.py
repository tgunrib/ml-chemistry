import os
import os.path as osp
import re
from typing import Callable, Optional

import pandas as pd
import torch
from drugtax import drugtax
from rdkit import Chem
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

#Convert the dataset into pytorch geometric InMemory Dataset. It was adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/aqsol.html#AQSOL
# and https://iwatobipen.wordpress.com/2021/09/24/attentive-fp-with-pyg-rdkit-pyg-pytorch_geometric-chemoinformatics/

def process_panda(r_data, data_list, pre_transform):
    smiles = r_data.SMILES
    ys = r_data.logS
    ys = ys if isinstance(ys, list) else [ys]
    y = torch.tensor(ys, dtype=torch.float).view(1, -1)
    data = from_smiles(smiles)
    data.y = y
    data.butina_cluster = r_data.butina_cluster
    data.descriptors = r_data[5:209].values
    if pre_transform is not None:
        data = pre_transform(data)
    data_list.append(data)




class Molecule(InMemoryDataset):
    def __init__(self, root_dir, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.root_dir = root_dir
        self.name = name
        # skip calling data
        super(Molecule, self).__init__(None, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root_dir, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root_dir, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        dataset = pd.read_csv(self.raw_paths[0], index_col=0)
        print(dataset.head())
        # Select the columns to min-max scale using column index slices
        cols_to_scale = dataset.columns[5:212]  # Select columns 'B' and 'C'

        # Define the min-max scaling function using lambda
        minmax_scale = lambda x: (x - x.min()) / (x.max() - x.min())

        # Apply the min-max scaling function to the selected columns using apply()
        dataset[cols_to_scale] = dataset[cols_to_scale].apply(minmax_scale)
        dataset.dropna(inplace=True,axis=1)
        print(dataset.head())
        data_list = []
        dataset.apply(lambda x : process_panda(x,data_list,self.pre_transform), axis=1)
        dataset.dropna()
        torch.save(self.collate(data_list), self.processed_paths[0])


    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))



class GenFeatures(object):
    def __init__(self):
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

        self.kingdoms = ['organic', 'inorganic']
        self.subclasses = [["organoheterocyclic", "organosulfur", "lipids", "allenes", "benzenoids",
                            "phenylpropanoids_and_polyketides",
                            "organic_acids", "alkaloids", "organic_salts", "organic_nitrogen", "organic_oxygen",
                            "organophosphorus",
                            "organohalogens", "organometallics", "nucleosides_nucleotides_analogues",
                            "lignans_neolignans_and_related",
                            "organic_polymers", "hydrocarbon_derivatives", "hydrocarbon", "organic_zwitterions",
                            "organic_cations",
                            "organic_anions", "carbenes", "organic_1_3_dipolar", "organopnictogens", "acetylides",
                            "homogenous_metal", "homogenous_non_metal", "mixed_metal_non_metal", "inorganic_salts",
                            "miscellaneous_inorganic"]]

        self.kingdom_label = LabelBinarizer()
        self.kingdom_label.fit(self.kingdoms)
        self.subclass_label = MultiLabelBinarizer()
        self.subclass_label.fit(self.subclasses)

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.
        mol = Chem.MolFromSmiles(data.smiles)
        info = drugtax.DrugTax(input_smile=data.smiles, input_type='string')
        subclasses = self.subclass_label.transform([info.superclasses])
        kingdom = self.kingdom_label.transform([[info.kingdom]])
        butina = data.butina_cluster

        xs = []
        xs_combined = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            if atom.GetSymbol() not in self.symbols:
                symbol[self.symbols.index('other')] = 1
            else:
                symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 6
            if atom.GetDegree() < 6:
                degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            if atom.GetHybridization not in self.hybridizations:
                hybridization[self.hybridizations.index('other')] = 1
            else:
                hybridization[self.hybridizations.index(
                    atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

            x_combined = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type + [butina] + list(kingdom[0]) + list(subclasses[0]))
            xs_combined.append(x_combined)

        data.x = torch.stack(xs, dim=0)
        data.x_combined = torch.stack(xs_combined,dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)
        print([data.descriptors])
        data.descriptors = torch.stack([torch.FloatTensor(list(data.descriptors) +
                                [int(butina)] + list(kingdom[0])+ list(subclasses[0]))], dim=0)
        data.no_descriptors = torch.stack([torch.FloatTensor([int(butina)] + list(kingdom[0])+ list(subclasses[0]))], dim=0)
        print(data)
        return data

import torch_geometric.transforms as T
transform = T.Compose([GenFeatures()])
dataset = Molecule('C:\\Users\\tosin\\Documents\\IP', 'features.csv',
                   pre_transform=transform)
test = Molecule('C:\\Users\\tosin\\Documents\\IP', 'esol.csv',
                   pre_transform=transform)
N = len(dataset) // 10
val_dataset = dataset[:N]
test_dataset = test
train_dataset = dataset[2 * N:]
