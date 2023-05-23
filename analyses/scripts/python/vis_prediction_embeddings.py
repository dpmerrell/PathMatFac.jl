
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib as mpl
import script_util as su
from os import path
import pandas as pd
import numpy as np
import argparse
import h5py

NAMES = su.NICE_NAMES

mpl.rcParams['font.family'] = "serif"

def hdfs_to_embedding(train_hdfs, test_hdfs):

    train_hdf = train_hdfs[0]
    train_X = su.load_hdf(train_hdf, "X", dtype=float).transpose()
    train_y = su.load_hdf(train_hdf, "target", dtype=str).transpose()
    
    test_hdf = test_hdfs[0]
    test_X = su.load_hdf(test_hdf, "X", dtype=float).transpose()
    test_y = su.load_hdf(test_hdf, "target", dtype=str).transpose()

    pca = PCA(n_components=2, whiten=True)
    pca.fit(train_X, train_y)

    train_X = pca.transform(train_X)
    test_X = pca.transform(test_X)
    
    return train_X, train_y, test_X, test_y



def plot_classifier_embedding(ax, X_train, y_train, X_test, y_test, color_list):

    labels = sorted(list(set(np.unique(y_train)).union(np.unique(y_test))))
    n_colors = len(color_list)
    for i, label in enumerate(labels):
        color = color_list[i % n_colors]

        train_rows = (y_train == label)
        ax.scatter(X_train[train_rows,0],
                   X_train[train_rows,1], 
                   color=color, marker="o", s=0.25)

        test_rows = (y_test == label)
#        ax.scatter(X_test[test_rows,0],
#                   X_test[test_rows,1], 
#                   color="red", marker="*", s=2.0)
#                   #color=color, marker="*", s=2.0)

    return


def plot_binary_embedding(ax, train_X, train_y, test_X, test_y):
    plot_classifier_embedding(ax, train_X, train_y, test_X, test_y, ["red","blue"])
    return


def plot_multiclass_embedding(ax, train_X, train_y, test_X, test_y):
    plot_classifier_embedding(ax, train_X, train_y, test_X, test_y, su.ALL_COLORS)
    return


def plot_regression_embedding(ax, train_X, train_y, test_X, test_y):
   
    train_y = train_y.astype(float)
    test_y = test_y.astype(float)

    vmin = min(np.min(train_y), np.min(test_y))
    vmax = max(np.max(train_y), np.max(test_y))

    ax.scatter(train_X[:,0], train_X[:,1],
               c=train_y, 
               vmin=vmin, vmax=vmax, cmap="binary",
               marker="o", s=0.25)

#    ax.scatter(test_X[:,0], test_X[:,1], 
#               color="red",
#               #c=test_y,
#               #vmin=vmin, vmax=vmax, cmap="binary",
#               marker="*", s=2.0)

    return


def plot_survival_embedding(ax, train_X, train_y, test_X, test_y):
    
    train_y = train_y.astype(float)
    test_y = test_y.astype(float)

    # Sift out the living and dead from the training set
    dead_train = np.isnan(train_y[:,1])
    alive_train = np.logical_not(dead_train)
    train_alive_X = train_X[alive_train,:]
    train_alive_y = train_y[alive_train,1]
    train_dead_X = train_X[dead_train,:]
    train_dead_y = train_y[dead_train,0]

    # Sift out the living and dead from the training set
    dead_test = np.isnan(test_y[:,1])
    alive_test = np.logical_not(dead_test)
    test_alive_X = test_X[alive_test,:]
    test_alive_y = test_y[alive_test,1]
    test_dead_X = test_X[dead_test,:]
    test_dead_y = test_y[dead_test,0]

    alive_vmin = np.quantile(np.concatenate((train_alive_y, test_alive_y)), 0.125)
    alive_vmax = np.quantile(np.concatenate((train_alive_y, test_alive_y)), 0.875)
    
    dead_vmin = np.quantile(np.concatenate((train_dead_y, test_dead_y)), 0.125)
    dead_vmax = np.quantile(np.concatenate((train_dead_y, test_dead_y)), 0.875)

    # (train, living)
    ax.scatter(train_alive_X[:,0], train_alive_X[:,1],
               c=train_alive_y, 
               vmin=alive_vmin, vmax=alive_vmax, cmap="Blues",
               marker="o", s=0.25)
 #   # (test, living)
 #   ax.scatter(test_alive_X[:,0], test_alive_X[:,1], 
 #              color="red",
 #              #c=test_alive_y,
 #              #vmin=alive_vmin, vmax=alive_vmax, cmap="Blues",
 #              marker="*", s=0.5)

    # (train, dead)
    ax.scatter(train_dead_X[:,0], train_dead_X[:,1],
               c=train_dead_y, 
               vmin=dead_vmin, vmax=dead_vmax, cmap="Reds_r",
               marker="o", s=0.25)
#    # (test, dead)
#    ax.scatter(test_dead_X[:,0], test_dead_X[:,1],
#               color="red", 
#               #c=test_dead_y,
#               #vmin=dead_vmin, vmax=dead_vmax, cmap="Reds_r",
#               marker="*", s=0.5)
    return


def plot_prediction_embeddings(ax, result_data): 
    """

    """

    i,j = result_data["idx"]
    nrow, ncol = result_data["N"]
    rowname, colname = result_data["names"]
    train_hdfs = result_data["train_hdfs"] 
    test_hdfs = result_data["test_hdfs"] 

    embedded_data = hdfs_to_embedding(train_hdfs, test_hdfs) 

    print(rowname, colname, embedded_data[0].shape, embedded_data[2].shape)

    if colname in ("hpv_status"):
        plot_binary_embedding(ax, *embedded_data)
    elif colname in ("ctype"):
        plot_multiclass_embedding(ax, *embedded_data)
    elif colname in ("survival"):
        plot_survival_embedding(ax, *embedded_data)
    elif colname in ("pathologic_stage"):
        embedded_data = list(embedded_data)
        embedded_data[1] = su.encode_pathologic_stage(embedded_data[1]) 
        embedded_data[3] = su.encode_pathologic_stage(embedded_data[3]) 
        plot_regression_embedding(ax, *embedded_data)

    if i == nrow - 1:
        ax.set_xlabel(NAMES[colname])
    if j == 0:
        ax.set_ylabel(NAMES[rowname])

    return 

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("out_png")
    parser.add_argument("--train_hdfs", nargs="+")
    parser.add_argument("--test_hdfs", nargs="+")

    args = parser.parse_args()
    train_hdfs = args.train_hdfs
    test_hdfs = args.test_hdfs
    out_png = args.out_png

    # Arrange the HDF5 filepaths into a grid,
    # indexed by (method, target)
    train_hdfs = su.get_methods_targets(train_hdfs)
    train_grid, method_names, target_names = su.dict_to_grid(train_hdfs)
    test_hdfs = su.get_methods_targets(test_hdfs)
    test_grid, _, _ = su.dict_to_grid(test_hdfs)
    
    # Add some auxiliary data to the grid
    nrow = len(method_names)
    ncol = len(target_names)
    grid = [[{"idx":(i,j), 
              "N": (nrow, ncol), 
              "names":(method_names[i],target_names[j]), 
              "train_hdfs": train_dat,
              "test_hdfs": test_grid[i][j]} for j, train_dat in enumerate(row)] for i, row in enumerate(train_grid)]

    method_names = [NAMES[mn] for mn in method_names]
    target_names = [NAMES[tn] for tn in target_names]
    
    # Plot prediction results across the grid
    fig, axarr = su.make_subplot_grid(plot_prediction_embeddings, grid, 
                                      method_names, target_names)

    #fig.text(0.5, 0.04, "Prediction targets", ha="center")
    #fig.text(0.04, 0.5, "Dimension reduction methods", rotation="vertical", ha="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

 
