

from collections import Counter, defaultdict
import script_util as su
import numpy as np
import argparse
import h5py


def load_data(data_hdf, clinical_hdf):

    omic_data = su.load_hdf(data_hdf, "omic_data/data").transpose()
    samples = su.load_hdf(data_hdf, "omic_data/instances", dtype=str)
    sample_groups = su.load_hdf(data_hdf, "omic_data/instance_groups", dtype=str)
    assays = su.load_hdf(data_hdf, "omic_data/feature_assays", dtype=str)
    genes = su.load_hdf(data_hdf, "omic_data/feature_genes", dtype=str)
    
    barcodes = su.load_hdf(data_hdf, "barcodes/data", dtype=str).transpose()
    barcode_assays = su.load_hdf(data_hdf, "barcodes/features", dtype=str)

    clinical_data = su.load_hdf(clinical_hdf, "data", dtype=str).transpose()
    clinical_samples = su.load_hdf(clinical_hdf, "samples", dtype=str)
    clinical_features = su.load_hdf(clinical_hdf, "features", dtype=str)

    return omic_data, samples, sample_groups, assays, genes, barcodes, barcode_assays, 
           clinical_data, clinical_samples, clinical_features


def idx_select(idx, *ls):
    return tuple(a[idx,...] for a in ls)


"""
    Filter out samples that have insufficient omic data
"""
def omic_filter(omic_data, omic_assays):

    # For now, we just remove the samples that don't have
    # enough useful mrnaseq measurements
    rnaseq_data = omic_data[:,omic_assays == "mrnaseq"]
    column_vars = np.nanvar(omic_data, axis=0)
    best_idx = np.argsort(column_vars)[-int(rnaseq_data.shape[1]*0.25):]
    rnaseq_data = rnaseq_data[:,best_idx]

    valid_rows = np.where(np.sum(np.isfinite(rnaseq_data), axis=1) > 50)
    
    return valid_rows 
    

def survival_col_filter(clinical_columns):
    dtd_idx = np.where(clinical_columns == "days_to_death")[0][0]
    dtlf_idx = np.where(clinical_columns == "days_to_last_followup")[0][0]
    return [dtd_idx, dtlf_idx]

def survival_row_filter(clinical_data, clinical_columns):
    days_to_death = clinical_data[:,np.where(clinical_columns == "days_to_death")[0][0]]
    t_final = clinical_data[:,np.where(clinical_columns == "days_to_last_followup")[0][0]]

    valid_rows = np.where( (days_to_death != "nan") | (t_final != "nan") )
    return np.arange(clinical_data.shape[0])


def default_col_filter(clinical_columns, filter_col):
    idx = np.where(clinical_columns == filter_col)[0][0]
    return [idx]

def default_row_filter(clinical_data, clinical_columns, filter_col):
    col_idx = np.where(clinical_columns == filter_col)[0][0]
    valid_rows = np.where(clinical_data[:,col_idx] != "nan")

    return valid_rows


def select_col_filter(target):
    if target == "survival":
        return survival_col_filter
    else:
        return lambda x: default_col_filter(x, target)

def select_row_filter(target):
    if target == "survival":
        return survival_row_filter
    else:
        return lambda x,y: default_row_filter(x,y,target)


def apply_filters(target_ls, omic_data, samples, sample_groups, 
                             assays, genes, barcodes, barcode_assays,
                             clinical_data, clinical_samples, clinical_features):

    # First, restrict ourselves to the intersection
    # of samples with omic and clinical data   
    omic_idx, clinical_idx = su.keymatch(samples, clinical_samples)

    # Once the omic and clinical data have identical order,
    # we can apply subsequent filters to them uniformly.
    to_filter = [omic_data, samples, sample_groups, barcodes]
    to_filter = list(idx_select(omic_idx, to_filter))
    clinical_data = clinical_data[clinical_idx]
    to_filter.append(clinical_data)

    # First, we filter out bad samples based on the omic data
    filter_idx = omic_filter(to_filter[0], assays)
    to_filter = idx_select(filter_idx, to_filter)

    # Apply row filters 
    for target in target_ls:

        # Get the corresponding row filter function;
        filter_fn = select_row_filter(target) 
        # Compute the filtered row indices;
        filter_idx = filter_fn(to_filter[-1], clinical_features)
        # and apply them to the samples of data
        to_filter = idx_select(filter_idx, to_filter) 

    filtered_omic, filtered_samples, filtered_groups, filtered_barcodes,\
    filtered_clinical = tuple(to_filter) 

    # Apply clinical data column filters
    clinical_idx = []
    for target in target_ls:
        filter_fn = select_col_filter(target)
        clinical_idx += filter_fn(clinical_features)
    
    clinical_features = clinical_features[clinical_idx] 
    filtered_clinical = filtered_clinical[clinical_idx]

    return filtered_omic, filtered_samples, filtered_groups,
           assays, genes, 
           filtered_barcodes, barcode_assays,
           filtered_clinical, clinical_features
           



def stratified_group_shuffle_split(class_labels, group_labels, split_fracs, srt_by_size=False):
    """
    A randomized algorithm for generating shuffled splits that 
    (1) are grouped by `group_labels` and 
    (2) are stratified by `class_labels`.

    The size/number of splits are defined by `split_fracs` (iterable)
    
    Assume discrete class labels.
    """
    split_fracs = np.array(split_fracs)
    
    # Get the unique classes and their occurrences
    cls_counts = Counter(class_labels)
    unq_classes = np.array(sorted(list(cls_counts.keys())))
    cls_encoder = {cls: idx for idx, cls in enumerate(unq_classes)}
    cls_counts = np.vectorize(lambda x: cls_counts[x])(unq_classes) 
    
    # Compute a grid of "capacities": ideal quantities 
    # of samples for each (group, class) pair.
    capacity = np.outer(split_fracs, cls_counts)
    
    # Get the unique groups. By default, sort them by decreasing size.
    # It makes the evaluation more challenging.
    unq_groups = np.random.permutation(np.unique(group_labels))
    if srt_by_size:
        gp_size_dict = Counter(group_labels)
        gp_sizes = [gp_size_dict[gp] for gp in unq_groups]
        gp_srt_idx = np.argsort(gp_sizes)[::-1]
        unq_groups = unq_groups[gp_srt_idx]
    
    # Collect information about the groups' samples and labels
    gp_encoder = {gp: idx for idx, gp in enumerate(unq_groups)}
    gp_vecs = np.zeros((len(unq_groups), len(unq_classes)))
    gp_to_samples = defaultdict(lambda : [])
    for idx, (gp, cls) in enumerate(zip(group_labels, class_labels)):
        gp_vecs[gp_encoder[gp], cls_encoder[cls]] += 1
        gp_to_samples[gp].append(idx)
    
    # We will assign groups to these splits
    split_sets = [set() for _ in split_fracs]
    
    # Randomly assign groups to splits
    for i in range(gp_vecs.shape[0]):
        # Randomization is weighted by (a) available capacity and
        # (b) the group's class distribution.
        gp_counts = gp_vecs[i,:]
        weight_vec = np.dot(capacity, gp_counts)
        split_idx = np.random.choice(len(split_fracs), p=weight_vec/np.sum(weight_vec))
        
        # Add group to split; decrement capacity
        split_sets[split_idx].add(unq_groups[i])
        capacity[split_idx,:] -= gp_counts
        capacity[capacity < 0] = 0
    
    split_idxs = [np.array(sorted(sum((gp_to_samples[gp] for gp in s), [])), dtype=int) for s in split_sets]
 
    # Return splits
    return split_idxs


def stratified_group_crossval_splits(class_labels, group_labels, n_folds=5):

    # Use the stratified grouped shuffle split to generate
    # `n_folds` roughly equally-sized splits. These will be
    # the cross-validation test sets.
    split_fracs = [1.0/n_folds]*n_folds    
    test_sets = stratified_group_shuffle_split(class_labels, group_labels, 
                                               split_fracs, srt_by_size=False)
 
    # Let their complements be the cross-validation training sets.
    train_sets = [np.concatenate(tuple(v for j, v in enumerate(test_sets) if j != i)) for i, _ in enumerate(test_sets)]

    # Zip the training and test sets together.
    cv_splits = list(zip(train_sets, test_sets))
 
    return cv_splits


def define_sample_groups(barcodes, barcode_assays):

    #M, _ = barcodes.shape
    #group_labels = np.arange(M, dtype=int)

    print("BARCODES: ", barcodes)
    print("BARCODE ASSAYS: ", barcode_assays)
    mrnaseq_idx = np.where(barcode_assays == "mrnaseq")[0][0]
    print("MRNASEQ IDX: ", mrnaseq_idx)
    mrnaseq_barcodes = barcodes[:,mrnaseq_idx]
    print("MRNASEQ BARCODES: ", mrnaseq_barcodes)
    group_labels = np.vectorize(lambda x: "-".join(x.split("-")[-2:]))(mrnaseq_barcodes) 
    print("GROUP LABELS: ", group_labels)

    return group_labels


def split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 targets, target_names, idx, output_path):

    with h5py.File(output_path, "w") as f:
        su.write_hdf(f, "omic_data/data", omic_data[idx,:].transpose())
        su.write_hdf(f, "omic_data/instances", samples[idx], is_string=True)
        su.write_hdf(f, "omic_data/instance_groups", ctype_labels[idx], is_string=True)
        su.write_hdf(f, "omic_data/feature_assays", assays, is_string=True)
        su.write_hdf(f, "omic_data/feature_genes", genes, is_string=True)

        su.write_hdf(f, "barcodes/data", barcodes[idx,:].transpose(), is_string=True)
        su.write_hdf(f, "barcodes/instances", samples[idx], is_string=True)
        su.write_hdf(f, "barcodes/features", barcode_assays, is_string=True)
        
        su.write_hdf(f, "targets/data", targets[idx,:].transpose(), is_string=True)
        su.write_hdf(f, "targets/features", target_names[idx], is_string=True)

    return


def save_split(omic_data, samples, ctype_labels,
               assays, genes, barcodes, barcode_assays,
               targets, target_names, train_idx, test_idx, output_path_prefix):

    split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 targets, target_names, train_idx, output_path_prefix+"__train.hdf")    
    split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 targets, target_names, test_idx, output_path_prefix+"__test.hdf") 
    return


def save_splits(omic_data, samples, ctype_labels,
                assays, genes, barcodes, barcode_assays,
                targets, target_names, splits, output_path_prefix):

    for i, (train_idx, test_idx) in enumerate(splits):
        pref = output_path_prefix + f"__fold={i}"
        save_split(omic_data, samples, ctype_labels,
                   assays, genes, barcodes, barcode_assays,
                   targets, target_names, train_idx, test_idx, pref)
    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("omic_hdf")
    parser.add_argument("clinical_hdf")
    parser.add_argument("output_prefix")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--sample_filters", nargs="*")
    
    args = parser.parse_args()
    data_hdf = args.full_data_hdf
    out_prefix = args.output_prefix
    n_folds = args.n_folds
    filters = args.sample_filters

    data_arrays = load_data(data_hdf, clinical_hdf)

    filtered_omic, filtered_samples,\
    filtered_groups,\
    assays, genes, \
    barcodes, \
    barcode_assays,\
    filtered_clinical,\
    filtered_clinical_features = apply_filters(filters, *data_arrays)

    split_groups = define_sample_groups(barcodes, barcode_assays)

    cross_val_splits = stratified_group_crossval_splits(ctype_labels, 
                                                        split_groups,
                                                        n_folds=n_folds)
    
    save_splits(filtered_omic, filtered_samples, filtered_groups,
                assays, genes, filtered_barcodes, barcode_assays,
                filtered_clinical, filtered_clinical_features, cross_val_splits, out_prefix)

    
