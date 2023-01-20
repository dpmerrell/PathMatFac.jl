
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import script_util as su
import pickle as pkl
import numpy as np
import argparse
import h5py
import json


def load_data(training_hdf):

    # Load the features and labels from the HDF file
    X = su.load_hdf(training_hdf, "X")
    labels = su.load_hdf(training_hdf, "target", dtype=str)

    # Encode the labels as integers.
    # (label --> its rank in sorted order)
    unq_labels = np.sort(np.unique(labels))
    encoder = {ct:i for i, ct in enumerate(unq_labels)}
    y = np.vectorize(lambda x: encoder[x])(labels)

    return X, y


def fit_classifier(X, y, **kwargs):

    model = RandomForestClassifier(**kwargs)
    model.fit(X,y)

    return model


def compute_scores(model, X, y):

    pred_y_probs = model.predict_proba(X)
    pred_y = model.predict(X)
   
    scores = {'f1_micro': f1_score(y, pred_y, average="micro"),
              'f1_macro': f1_score(y, pred_y, average="macro"),
             }
    
    if pred_y_probs.shape[1] > 2:
        scores['roc_auc_ovr'] = roc_auc_score(y, pred_y_probs, multi_class="ovr")
        scores['roc_auc_ovo'] = roc_auc_score(y, pred_y_probs, multi_class="ovo")
    else:
        scores['roc_auc'] = roc_auc_score(y, pred_y_probs[:,1])

    return scores


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training_hdf")
    parser.add_argument("fitted_pkl")
    parser.add_argument("scores_json")

    args = parser.parse_args()
    training_hdf = args.training_hdf
    fitted_pkl = args.fitted_pkl
    scores_json = args.scores_json

    X, y = load_data(training_hdf)
    model = fit_classifier(X, y)
    pkl.dump(model, open(fitted_pkl, "wb"))

    score_dict = compute_scores(model, X, y)
    json.dump(score_dict, open(scores_json, "w"))


