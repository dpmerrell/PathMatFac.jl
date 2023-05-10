
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score 
import script_util as su
import pickle as pkl
import numpy as np
import argparse
import h5py
import json


def load_data(training_hdf):

    # Load the features and labels from the HDF file
    X = su.load_hdf(training_hdf, "X").transpose()
    y = su.load_hdf(training_hdf, "target", dtype=str)

    return X, y


def fit_regressor(X, y, **kwargs):

    model = RandomForestRegressor(**kwargs)
    model.fit(X,y)

    return model


def compute_scores(model, X, y):
    pred_y = model.predict(X)
    scores = {'mse': mean_squared_error(y, pred_y),
              'r2': r2_score(y, pred_y)
             }
    return scores


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training_hdf")
    parser.add_argument("fitted_pkl")
    parser.add_argument("scores_json")
    parser.add_argument("--target", default="pathologic_stage")

    args = parser.parse_args()
    training_hdf = args.training_hdf
    fitted_pkl = args.fitted_pkl
    scores_json = args.scores_json
    target = args.target

    X, y = load_data(training_hdf)
    if target == "pathologic_stage":
        y = su.encode_pathologic_stage(y)

    rf_kwargs = {"n_estimators": 500}

    model = fit_regressor(X, y, **rf_kwargs)
    pkl.dump(model, open(fitted_pkl, "wb"))

    score_dict = compute_scores(model, X, y)
    json.dump(score_dict, open(scores_json, "w"))


