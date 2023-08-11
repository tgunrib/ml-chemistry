import rdkit.Chem
from rdkit.Chem import rdFingerprintGenerator
from re import split

import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import drugtax
from sklearn.preprocessing import MultiLabelBinarizer


# This code calculate the tanimoto distance and fingerprints. It has been adapted from  url: https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html

def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix


def cluster_fingerprints(fingerprints, cutoff=0.2):
    """Cluster fingerprints
    Parameters:
        fingerprints
        cutoff: threshold for the clustering
    """
    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def convert_cluster_list(clusters):
    output = []
    for i in range(0, len(clusters)):
        for nums in clusters[i]:
            output.insert(nums, i)
    return output


def drug_tax(smiles, filename):
    data = []
    for smile in smiles:
        try:
            taxonomy = drugtax.DrugTax(smile, input_type='string')
            kingdom = taxonomy.kingdom
            classes = taxonomy.superclasses
            data.append((smile, classes, kingdom))
        except Exception as e:
            print(e)
    df = pd.DataFrame(data, columns=['SMILES', 'classes', 'kingdom'])
    df.dropna().to_csv(filename)
    return df.dropna()


if __name__ == '__main__':
    data = pd.read_csv('../raw/esol.csv', index_col=0)
    drugtax = drug_tax(list(data['SMILES']), '../raw/taxonomy_esol')

    subclasses = [["organoheterocyclic", "organosulfur", "lipids", "allenes", "benzenoids",
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

    mlb = MultiLabelBinarizer()
    mlb.fit_transform(subclasses)

    res = pd.DataFrame(mlb.transform(drugtax.classes),
                       columns=mlb.classes_,
                       index=drugtax.index)

    result = pd.merge(drugtax, res, left_index=True, right_index=True)
    result.to_csv('raw/taxonomy_features')
    merged_df = pd.merge(data, result,  how="inner", on=["SMILES"])
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(5)
    fingerprints = [rdkit_gen.GetFingerprint(rdkit.Chem.MolFromSmiles(x)) for x in merged_df.SMILES]
    clusters = cluster_fingerprints(fingerprints, cutoff=0.45)
    cluster_values = convert_cluster_list(clusters)
    merged_df['butina_cluster'] = cluster_values
    print(len(merged_df['butina_cluster']))
    merged_df.to_csv('raw/features_esol.csv',header=True, index=True)
