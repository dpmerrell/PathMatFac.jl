
import script_util as su
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_offset(values, feature_or_instance):
    plt.plot(values)

if __name__=="__main__":

    args = sys.argv[1:]
    model_hdf = args[0]
    feature_or_instance = args[1]
    out_png = args[2]

    if feature_or_instance == "feature":
        original_genes,\
        original_assays,\
        augmented_genes,\
        feat_to_idx = su.load_feature_info(model_hdf)

        offset = su.load_feature_offset(model_hdf)

    elif feature_or_instance == "instance":
        original_samples,\
        original_groups,\
        augmented_samples,\
        sample_to_idx = su.load_sample_info(model_hdf)
        
        offset = su.load_instance_offset(model_hdf)

    plt.figure(figsize=(10,5))

    plot_offset(offset, feature_or_instance)

    title_dict = {"feature": "Feature offset",
                  "instance": "Instance offset"
                  }

    title = title_dict[feature_or_instance]
    plt.title(title)

    plt.savefig(out_png, dpi=400)

