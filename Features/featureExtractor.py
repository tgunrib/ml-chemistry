import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.preprocessing import MultiLabelBinarizer

from featuriser import extract_data, merge_data, canonical_smiles, rdkit_descriptor, rdkit_fingerprints
from GCN_Files.featuriser_graph import create_pytorch_geometric_graph_data_list_from_smiles_and_labels

# This adds graphs and classification given a dataset used for modelling
class FeatureExtractor:

    def __init__(self, list_files, filename):
        all_data = []
        for file in list_files:
            all_data.append(extract_data(file[0], file[1]))
        data = merge_data(all_data)
        data["SMILES"] = canonical_smiles(data['SMILES'])
        # Create a list for duplicate smiles
        duplicates_smiles = data[data['SMILES'].duplicated()]['SMILES'].values
        len(duplicates_smiles)
        data[data['SMILES'].isin(duplicates_smiles)].sort_values(by=['SMILES'])
        dataset_new = data.drop_duplicates(subset=['SMILES'])
        print(len(dataset_new))
        PandasTools.AddMoleculeColumnToFrame(dataset_new, 'SMILES', 'MOLECULES')
        print(dataset_new.columns)
        self.dataset = dataset_new

        # Extract the molecules for the descriptors
        descriptors = rdkit_descriptor(list(self.dataset.MOLECULES))
        # Descriptors
        df_with_200_descriptors = pd.DataFrame(descriptors[0], columns=descriptors[1])
        self.dataset = pd.merge(self.dataset, df_with_200_descriptors, left_index=True, right_index=True)

        # Fingerprints
        fingerprints = rdkit_fingerprints(list(self.dataset.MOLECULES))
        df_with_fingerprints = pd.DataFrame(fingerprints[0], columns=fingerprints[1])
        self.dataset = pd.merge(self.dataset, df_with_fingerprints, left_index=True, right_index=True)

        x_smiles = list(self.dataset.SMILES)
        y = list(self.dataset.logS)

        # create list of molecular graph objects from list of SMILES x_smiles and list of labels y
        data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)
        self.dataset['Graph'] = data_list


        # save dataframe
        self.dataset = self.dataset.dropna()
        self.dataset = self.dataset.reset_index(drop=True)
        self.dataset.to_csv(filename + '.csv', header=True)

    def get_dataset(self):
        return self.dataset


if __name__ == '__main__':
    filenames = [('Datasets/water_solubility_data.csv', 'LogS')]
    """ ('Datasets/ESOL.csv', 'measured log(solubility:mol/L)'),
                ('Datasets/AQSOLDB.csv', 'Solubility')
                 ('Datasets/Improved Prediction of Aqueous Solubility of Novel Compounds by Going Deeper With Deep '
                  'Learning.csv', 'logS'),
                 ('Datasets/Multi-channel GCN dataset.csv', 'Experimental Solubility in Water'),
                 ('Datasets/OLS_Lasso_High-temperature-water.csv', 'logS(mol/kg)',) """


    features = FeatureExtractor(list_files=filenames, filename='../raw/esol.csv')

