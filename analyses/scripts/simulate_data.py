
import argparse
from scipy import sparse 
import numpy as np
import json
import h5py

# Get the set of all vertices from all pathways.
# Make sure they're in a canonical order.


# translate each pathway into a precision matrix


# translate the patient hierarchy into a precision matrix
# (i.e., graph Laplacian)


# Simulate the data!

    # Sample a feature profile for each pathway
    # precision matrices == pathway precision matrices

    # Sample K patient profiles (1 for each pathway)
    # precision matrix == patient precision matrix

    # matrix multiplication + Gaussian noise

    # return the data *and* the linear factors.


# save to an HDF file
    # Tables for the data, rownames, column names, 
    # *and* for the linear factors.


if __name__=="__main__":


    parser = argparse.ArgumentParser("Simulate a dataset, parameterized by a set of pathways and a patient hierarchy")
    parser.add_argument("pathway_json", help="JSON file containing our pathways")
    parser.add_argument("patient_hierarchy_json", help="JSON file containing a hierarchy of patients")

    args = parser.parse_arguments()

    # read a pathway file
    with open(args.pathway_json, "r") as f_pathways:
        pathways = json.load(f_pathways)

    # read a patient hierarchy file
    with open(args.patient_hierarchy_json, "r") as f_patients:
        patients = json.load(f_patients)

    # get the full set of features
    feature_list = sorted(pathways["all_entities"])

    # Sample feature profiles (MxK) 

    # Sample patient profiles (KxN)

    # simulate the data

    # write to file
