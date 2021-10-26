import h5py
import numpy as np
import argparse
import sys


def parse_feature_names(feature_names):

    feature_genes = np.vectorize(lambda x: x.split("_")[0])(feature_names)
    feature_assays = np.vectorize(lambda x: x.split("_")[-1])(feature_names)

    return feature_genes, feature_assays


def load_data(input_hdf):

    with h5py.File(input_hdf, "r") as f:

        omic_matrix = f["data"][:,:]
        sample_ids = f["instances"][:].astype(str)
        sample_groups = f["cancer_types"][:].astype(str)
        feature_names = f["features"][:].astype(str)

    feature_genes, feature_assays = parse_feature_names(feature_names)


    return omic_matrix, sample_ids, sample_groups, feature_genes, feature_assays


def hdf_write_string_vector(f_out, path, array):
    dataset = f_out.create_dataset(path, shape=array.shape,
                                   dtype=h5py.string_dtype("utf-8"))
    dataset[:] = array
    return


def output_to_hdf(output_hdf, omic_matrix, sample_ids, sample_groups, 
                                           feature_genes, feature_assays):

    with h5py.File(output_hdf, "w") as f:
        hdf_write_string_vector(f, "instances", sample_ids)
        hdf_write_string_vector(f, "instance_groups", sample_groups)
        hdf_write_string_vector(f, "feature_genes", feature_genes)
        hdf_write_string_vector(f, "feature_assays", feature_assays)
       
        dataset = f.create_dataset("data", shape=omic_matrix.shape,
                                           dtype=float)
        dataset[:,:] = omic_matrix

    return


def inv_logistic(a):
    return np.log(a / (1.0 - a))


def cna_threshold(a, l=-0.5, u=0.5):
    l_idx = (a <= l)
    u_idx = (a > u)
    mid_idx = ( np.logical_not(l_idx) & np.logical_not(u_idx) )

    a[l_idx] = 0.0
    a[u_idx] = 1.0
    a[mid_idx] = 0.5

    return a


def mut_threshold(a, u=0.0):
    u_idx = (a > u)
    a[u_idx] = 1.0
    return a


def preprocess_features(omic_matrix, feature_assays):

    methylation_rows = (feature_assays == "methylation")
    cna_rows = (feature_assays == "cna")
    mut_rows = (feature_assays == "mutation")

    omic_matrix[methylation_rows,:] = inv_logistic(omic_matrix[methylation_rows,:])
    print("Transformed methylation data")
    omic_matrix[cna_rows,:] = cna_threshold(omic_matrix[cna_rows,:])
    print("Transformed CNA data")
    omic_matrix[mut_rows,:] = mut_threshold(omic_matrix[mut_rows,:])
    print("Transformed Mutation data")

    return omic_matrix


if __name__=="__main__":

    args = sys.argv
    
    input_hdf = args[1]
    output_hdf = args[2]

    omic_matrix,\
    sample_ids, sample_groups, \
    feature_genes, feature_assays = load_data(input_hdf)

    print(omic_matrix.shape)
    print(sample_ids.shape)
    print(feature_genes.shape)

    prepped_omics = preprocess_features(omic_matrix, feature_assays)
    
    output_to_hdf(output_hdf, prepped_omics, 
                              sample_ids, sample_groups,
                              feature_genes, feature_assays)

    

