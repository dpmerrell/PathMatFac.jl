

from collections import Counter, defaultdict
import script_util as su
import numpy as np
import argparse
import h5py


rng = np.random.default_rng(12345)


def load_data(data_hdf, clinical_hdf):

    omic_data = su.load_hdf(data_hdf, "omic_data/data").transpose()
    samples = su.load_hdf(data_hdf, "omic_data/instances", dtype=str)
    sample_groups = su.load_hdf(data_hdf, "omic_data/instance_groups", dtype=str)
    assays = su.load_hdf(data_hdf, "omic_data/feature_assays", dtype=str)
    genes = su.load_hdf(data_hdf, "omic_data/feature_genes", dtype=str)
    
    barcodes = su.load_hdf(data_hdf, "barcodes/data", dtype=str).transpose()
    barcode_assays = su.load_hdf(data_hdf, "barcodes/features", dtype=str)

    clinical_data = su.load_hdf(clinical_hdf, "data", dtype=h5py.string_dtype("utf-8")).transpose()
    clinical_samples = su.load_hdf(clinical_hdf, "columns", dtype=str)
    clinical_features = su.load_hdf(clinical_hdf, "index", dtype=str)

    return omic_data, samples, sample_groups, assays, genes, barcodes,\
           barcode_assays, clinical_data, clinical_samples, clinical_features


def idx_select(idx, *ls):
    return [a[idx,...] for a in ls]


"""
    Filter out samples that have insufficient omic data
"""
def omic_filter(omic_data, omic_assays):

    # For now, we just remove the samples that don't have
    # enough useful mrnaseq measurements
    rnaseq_data = omic_data[:,omic_assays == "mrnaseq"]
    rnaseq_data = rnaseq_data[:,np.sum(np.isfinite(rnaseq_data), axis=0) > 50]

    column_vars = np.nanvar(rnaseq_data, axis=0)
    srt_idx = np.argsort(column_vars)
    best_idx = srt_idx[-int(rnaseq_data.shape[1]*0.05):]

    rnaseq_data = rnaseq_data[:,best_idx]

    valid_rows = np.where(np.sum(np.isfinite(rnaseq_data), axis=1) > 50)
    return valid_rows[0] 
    


def survival_data_selector(features, data, ctypes):
    dtd_idx = np.where(features == "days_to_death")[0][0]
    dtlf_idx = np.where(features == "days_to_last_followup")[0][0]
    return data[:,[dtd_idx,dtlf_idx]]


def survival_row_filter(clinical_data, clinical_columns):
    days_to_death = clinical_data[:,np.where(clinical_columns == "days_to_death")[0][0]]
    t_final = clinical_data[:,np.where(clinical_columns == "days_to_last_followup")[0][0]]

    valid_rows = np.where( (days_to_death != b"nan") | (t_final != b"nan") )[0]
    return valid_rows


def default_col_filter(features, data, target):
    idx = np.where(features == target)[0][0]
    return data[:,idx]


def default_row_filter(clinical_data, clinical_columns, filter_col):
    col_idx = np.where(clinical_columns == filter_col)[0][0]
    valid_rows = np.where((clinical_data[:,col_idx] != b"nan") & (clinical_data[:,col_idx] != b"indeterminate"))[0]

    return valid_rows

def id_row_filter(clinical_data, clinical_columns):
    return np.arange(clinical_data.shape[0])

def target_data_selector(target):
    if target == "ctype":
        return lambda feat, dat, ctypes: ctypes
    if target == "survival":
        return survival_data_selector
    else:
        return lambda feat, dat, ctypes: default_col_filter(feat, dat, target)

def select_row_filter(target):
    if target == "ctype":
        return id_row_filter
    if target == "survival":
        return survival_row_filter
    else:
        return lambda x,y: default_row_filter(x,y,target)


def apply_filters(target, omic_data, samples, sample_groups, 
                          assays, genes, barcodes, barcode_assays,
                          clinical_data, clinical_samples, clinical_features):

    # First, restrict ourselves to the intersection
    # of samples with omic and clinical data   
    omic_idx, clinical_idx = su.keymatch(samples, clinical_samples)

    # Once the omic and clinical data have identical order,
    # we can apply subsequent filters to them uniformly.
    to_filter = [omic_data, samples, sample_groups, barcodes]
    to_filter = idx_select(omic_idx, *to_filter)
    clinical_data = clinical_data[clinical_idx,:]
    to_filter.append(clinical_data)

    # Filter out samples that don't have target data
    print("TARGET: ", np.unique(target))
    # Get the corresponding row filter function;
    filter_fn = select_row_filter(target) 
    # Compute the filtered row indices;
    filter_idx = filter_fn(to_filter[-1], clinical_features)
    # and apply them to the samples of data
    to_filter = idx_select(filter_idx, *to_filter) 

    # Next: filter out samples that don't have enough omic data
    filter_idx = omic_filter(to_filter[0], assays)
    to_filter = idx_select(filter_idx, *to_filter)

    filtered_omic, filtered_samples, filtered_groups, filtered_barcodes,\
    filtered_clinical = tuple(to_filter) 

    # Select the regression target data
    target_data = target_data_selector(target)(clinical_features, filtered_clinical, sample_groups)
 
    return filtered_omic, filtered_samples, filtered_groups,\
           assays, genes,\
           filtered_barcodes, barcode_assays,\
           target_data
           

def stratified_group_shuffle_split(class_labels, group_labels, split_fracs, max_tries=10):
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
  
    # Collect information about the groups' samples and labels
    unq_groups = rng.permutation(np.unique(group_labels))
    gp_encoder = {gp: idx for idx, gp in enumerate(unq_groups)}
    gp_vecs = np.zeros((len(unq_groups), len(unq_classes)))
    gp_to_samples = defaultdict(lambda : [])
    for idx, (gp, cls) in enumerate(zip(group_labels, class_labels)):
        gp_vecs[gp_encoder[gp], cls_encoder[cls]] += 1
        gp_to_samples[gp].append(idx)

    # Repeat this until we get a valid split 
    for _ in range(max_tries): 
        # Compute a grid of "capacities": ideal quantities 
        # of samples for each (split, class) pair.
        capacity = np.outer(split_fracs, cls_counts)
        orig_capacity = capacity[:,:]
 
        # We will assign groups to these splits
        split_sets = [set() for _ in split_fracs]
        
        # Randomly assign groups to splits
        for i in range(gp_vecs.shape[0]):
            # Randomization is weighted by (a) available capacity and
            # (b) the group's class distribution.
            gp_counts = gp_vecs[i,:]
            weight_vec = np.dot(capacity, gp_counts)
            split_idx = rng.choice(len(split_fracs), p=weight_vec/np.sum(weight_vec))
            
            # Add group to split; decrement capacity
            split_sets[split_idx].add(unq_groups[i])
            capacity[split_idx,:] -= gp_counts
            capacity[capacity < 0] = 0

        # Check that there's at least one sample assigned
        # to each (split, class) pair 
        if np.all(orig_capacity != capacity):
            break 

    split_idxs = [np.array(sorted(sum((gp_to_samples[gp] for gp in s), [])), dtype=int) for s in split_sets]

    # Return splits
    return split_idxs


def print_split_summary(split_idxs, class_labels, gp_labels):

    n_groups = len(np.unique(gp_labels))

    for i, (train_idx, test_idx) in enumerate(split_idxs):
        train_labels = class_labels[train_idx]
        train_gps = gp_labels[train_idx]
        test_labels = class_labels[test_idx]
        test_gps = gp_labels[test_idx]       

        unq_tr_l, unq_tr_l_c = np.unique(train_labels, return_counts=True)
        train_label_printout = " ".join([f"{l} ({c})" for (l, c) in zip(unq_tr_l, unq_tr_l_c)])
        
        unq_test_l, unq_test_l_c = np.unique(test_labels, return_counts=True)
        test_label_printout = " ".join([f" {l} ({c});" for (l, c) in zip(unq_test_l, unq_test_l_c)])

        n_gp_train = len(np.unique(train_gps))
        n_gp_test = len(np.unique(test_gps))

        print(f"Fold {i}:| N_train={len(train_labels)}: {train_label_printout} | N_test={len(test_labels)}: {test_label_printout}") 
        print(f"           train groups={n_gp_train}/{n_groups} | test groups = {n_gp_test}/{n_groups}")


def stratified_group_crossval_splits(class_labels, group_labels, n_folds=5):

    # Use the stratified grouped shuffle split to generate
    # `n_folds` roughly equally-sized splits. These will be
    # the cross-validation test sets.
    split_fracs = [1.0/n_folds]*n_folds    
    test_sets = stratified_group_shuffle_split(class_labels, group_labels, 
                                               split_fracs)
 
    # Let their complements be the cross-validation training sets.
    train_sets = [np.concatenate(tuple(v for j, v in enumerate(test_sets) if j != i)) for i, _ in enumerate(test_sets)]

    # Zip the training and test sets together.
    cv_splits = list(zip(train_sets, test_sets))
 
    print_split_summary(cv_splits, class_labels, group_labels)
 
    return cv_splits


def survival_target_stratification(target_data):

    labels = np.zeros(target_data.shape[0])

    dead_times = target_data[:,0].astype(float)
    dead_med = np.nanmedian(dead_times)
    labels[dead_times <= dead_med] = 0
    labels[dead_times > dead_med] = 1

    alive_times = target_data[:,1].astype(float)
    alive_med = np.nanmedian(alive_times)
    labels[alive_times <= alive_med] = 2
    labels[alive_times > alive_med] = 3
    
    return labels 


def target_stratification_labels(target, target_data):
    
    strat_target = target_data
    if target == "survival":
        strat_target = survival_target_stratification(target_data)
    if target == "pathologic_stage":
        strat_target = su.encode_pathologic_stage(target_data)

    return strat_target


def define_sample_groups(groupby, target, ctypes, barcodes, barcode_assays):

    group_labels = None
    if groupby == "batch":
        # For now, group by batch rather than cancer type
        mrnaseq_idx = np.where(barcode_assays == "mrnaseq")[0][0]
        mrnaseq_barcodes = barcodes[:,mrnaseq_idx]
        group_labels = np.vectorize(lambda x: "-".join(x.split("-")[-2:]))(mrnaseq_barcodes) 
    else:
        group_labels = ctypes[:]

    return group_labels


def split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 target_data, idx, output_path):

    print("OUTPUT PATH")
    print(output_path)
    with h5py.File(output_path, "w", driver='core') as f:
        su.write_hdf(f, "omic_data/data", omic_data[idx,:].transpose())
        su.write_hdf(f, "omic_data/instances", samples[idx], is_string=True)
        su.write_hdf(f, "omic_data/instance_groups", ctype_labels[idx], is_string=True)
        su.write_hdf(f, "omic_data/feature_assays", assays, is_string=True)
        su.write_hdf(f, "omic_data/feature_genes", genes, is_string=True)

        su.write_hdf(f, "barcodes/data", barcodes[idx,:].transpose(), is_string=True)
        su.write_hdf(f, "barcodes/instances", samples[idx], is_string=True)
        su.write_hdf(f, "barcodes/features", barcode_assays, is_string=True)
        
        su.write_hdf(f, "target", target_data[idx,...].transpose(), is_string=True)

    return


def save_split(omic_data, samples, ctype_labels,
               assays, genes, barcodes, barcode_assays,
               target_data, train_idx, test_idx, output_path_prefix):


    split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 target_data, train_idx, output_path_prefix+"__train.hdf")    
    split_to_hdf(omic_data, samples, ctype_labels,
                 assays, genes, barcodes, barcode_assays,
                 target_data, test_idx, output_path_prefix+"__test.hdf") 
    return


def save_splits(omic_data, samples, ctype_labels,
                assays, genes, barcodes, barcode_assays,
                target_data, target, splits, output_path_prefix):

    for i, (train_idx, test_idx) in enumerate(splits):
        pref = output_path_prefix + f"fold={i}" 
        save_split(omic_data, samples, ctype_labels,
                   assays, genes, barcodes, barcode_assays,
                   target_data, train_idx, test_idx, pref)
    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("omic_hdf")
    parser.add_argument("clinical_hdf")
    parser.add_argument("paradigm_hdf")
    parser.add_argument("output_prefix")
    parser.add_argument("--target")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--groupby", type=str, default="batch")
    parser.add_argument("--paradigm_filter", action="store_true", default=False)
 
    args = parser.parse_args()
    data_hdf = args.omic_hdf
    clinical_hdf = args.clinical_hdf
    paradigm_hdf = args.paradigm_hdf
    out_prefix = args.output_prefix
    target = args.target
    n_folds = args.n_folds
    groupby = args.groupby
    paradigm_filter = args.paradigm_filter

    print("PREFIX")
    print(out_prefix)

    # Load the omic and clinical data
    data_arrays = load_data(data_hdf, clinical_hdf)

    # Keep only samples with prediction target data
    filtered_omic, filtered_samples,\
    filtered_ctype_labels,\
    assays, genes, \
    barcodes, \
    barcode_assays,\
    target_data = apply_filters(target, *data_arrays)

    # Keep only samples with PARADIGM data
    if paradigm_filter:
        paradigm_samples = su.load_hdf(paradigm_hdf, "instances", dtype=str)
        left_idx, _ = su.keymatch(filtered_samples, paradigm_samples)
        filtered_omic, filtered_samples,\
        filtered_ctype_labels,\
        barcodes, target_data = idx_select(left_idx, filtered_omic, 
                                                     filtered_samples,
                                                     filtered_ctype_labels,
                                                     barcodes, target_data) 

    strat_target_data = target_stratification_labels(target, target_data)
    unq_targets, cnt = np.unique(strat_target_data, return_counts=True)
    print(unq_targets, " ", cnt)
    split_groups = define_sample_groups(groupby, target, filtered_ctype_labels, barcodes, barcode_assays)

    cross_val_splits = stratified_group_crossval_splits(strat_target_data, 
                                                        split_groups,
                                                        n_folds=n_folds)
    
    save_splits(filtered_omic, filtered_samples, filtered_ctype_labels,
                assays, genes, barcodes, barcode_assays,
                target_data, target, cross_val_splits, out_prefix)

    
