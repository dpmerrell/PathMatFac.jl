
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import script_util as su
import pickle as pkl
import numpy as np
import argparse
import json
import h5py


def load_data(test_hdf):

    # Load the features and labels from the HDF file
    X = su.load_hdf(test_hdf, "X").transpose()
    labels = su.load_hdf(test_hdf, "target", dtype=str)

    # Encode the labels as integers.
    # (label --> its rank in sorted order)
    unq_labels = np.sort(np.unique(labels))
    encoder = {lab:i for i, lab in enumerate(unq_labels)}
    y = np.vectorize(lambda x: encoder[x])(labels)

    return X, y, unq_labels


def compute_scores(y, pred_y, pred_y_probs, y_train):

    scores = {'f1_micro': f1_score(y, pred_y, average="micro"),
              'f1_macro': f1_score(y, pred_y, average="macro"),
              'accuracy': accuracy_score(y, pred_y),
             }
   
    # Distinguish between multiclass and binary
    if pred_y_probs.shape[1] > 2:
        scores['roc_auc_ovr'] = roc_auc_score(y, pred_y_probs, multi_class="ovr")
        scores['roc_auc_ovo'] = roc_auc_score(y, pred_y_probs, multi_class="ovo")
    else:
        scores['roc_auc'] = roc_auc_score(y, pred_y_probs[:,1])
    
    # Compute the performance of a trivial baseline
    unq_train, train_counts = np.unique(y_train, return_counts=True)
    trivial_pred = unq_train[np.argmax(train_counts)]
    test_counts = dict(zip(np.unique(y, return_counts=True)))
    trivial_accuracy = test_counts[trivial_pred] / len(y) 

    scores["accuracy_baseline"] = trivial_accuracy
    scores["roc_auc_baseline"] = 0.5

    return scores


def compute_other_attributes(y_labels, y, pred_y, pred_y_probs):

    results = {"classes": list(y_labels)}
    
    results["confusion"] = confusion_matrix(y,pred_y).tolist() 

    # Produce an ROC curve if this is binary classification
    if pred_y_probs.shape[1] == 2: 
        fpr, tpr, thresholds = roc_curve(y, pred_y_probs[:,1])
        results["roc"] = {"fpr": list(fpr),
                          "tpr": list(tpr)
                         }

    return results


if __name__=="__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pkl")
    parser.add_argument("train_hdf")
    parser.add_argument("test_hdf")
    parser.add_argument("score_json")
    parser.add_argument("other_output_json")

    args = parser.parse_args()

    model_pkl = args.model_pkl
    train_hdf = args.train_hdf
    test_hdf = args.test_hdf
    score_json = args.score_json
    other_output_json = args.other_output_json

    # Load model
    model = pkl.load(open(model_pkl, "rb"))
 
    # Load data
    _, y_train, _ = load_data(train_hdf)
    X, y, y_labels = load_data(test_hdf)

    # Make predictions
    y_pred_probs = model.predict_proba(X)
    y_pred = model.predict(X)

    # Score predictions
    score_dict = compute_scores(y, y_pred, y_pred_probs, y_train)

    # Output scores to JSON
    json.dump(score_dict, open(score_json, "w")) 

    # Compute other attributes of the prediction task
    # (E.g., confusion matrices or ROC curves)
    other_output = compute_other_attributes(y_labels, y, y_pred, y_pred_probs)
    json.dump(other_output, open(other_output_json, "w")) 


