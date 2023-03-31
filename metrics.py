#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from s_dbw import S_Dbw


def compute_distance_list(X, eval_distance_metric='cosine'):
    return spatial.distance.pdist(X, eval_distance_metric)


def metric_neighborhood_hit(X, y, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))


def calculate_cluster_score_diff(X_high, X_low, y, score_func, additional_params=None):
    if additional_params is None:
        additional_params = {}
    original_score = score_func(X_high, y, **additional_params)
    projection_score = score_func(X_low, y, **additional_params)
    return calculate_projection_metric_diff(original_score, projection_score)


def calculate_projection_metric_diff(original_score, projection_score):
    diff = original_score - projection_score
    return diff


def metric_trustworthiness(X_high, X_low, D_high_m, D_low_m, k=7):
    D_high = spatial.distance.squareform(D_high_m)
    D_low = spatial.distance.squareform(D_low_m)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_continuity(X_high, X_low, D_high_l, D_low_l, k=7):
    D_high = spatial.distance.squareform(D_high_l)
    D_low = spatial.distance.squareform(D_low_l)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]


def metric_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)


def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))


def metric_distance_consistency(X_2d, y):
    clf = NearestCentroid()
    clf.fit(X=X_2d, y=y)
    nearest_centroids = clf.predict(X=X_2d)
    num_same_label = sum([1 if y[i] == nearest_centroids[i] else 0 for i in range(len(y))])
    return num_same_label / len(y)


def compute_all_metrics(X, X_2d, D_high, D_low, y, eval_distance_metric='cosine'):
    T = metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metric_continuity(X, X_2d, D_high, D_low)
    R = metric_shepard_diagram_correlation(D_high, D_low)
    S = metric_normalized_stress(D_high, D_low)
    N = metric_neighborhood_hit(X_2d, y)

    calinski_harabaz_low = calinski_harabasz_score(X_2d, y)
    silhouette_low = silhouette_score(X_2d, y, metric='euclidean')
    davies_bouldin_low = davies_bouldin_score(X_2d, y)
    # For documentation see: https://github.com/alashkov83/S_Dbw
    sdbw_low = S_Dbw(X_2d, y, centers_id=None, method='Kim', alg_noise='sep', centr='mean', nearest_centr=True,
                       metric='euclidean')
    dsc_low = metric_distance_consistency(X_2d, y)

    return T, C, R, S, N, calinski_harabaz_low, silhouette_low, davies_bouldin_low, sdbw_low, dsc_low


def evaluate_layouts(results_path, x, y, layouts_dict, dict_of_hyperparameter_dicts, old_df=None,
                     res_file_name="results", topic_level_experiment_name="", use_bow_for_comparison=True,
                     eval_distance_metric='cosine'):
    results = []
    if use_bow_for_comparison:
        high_distance_path = os.path.join(results_path, "high_distances_bow")
    else:
        high_distance_path = os.path.join(results_path, "high_distances_" + topic_level_experiment_name)

    if os.path.isfile(high_distance_path + ".npy"):
        D_high = np.load(high_distance_path + ".npy")
    else:
        D_high = compute_distance_list(x, eval_distance_metric=eval_distance_metric)
        np.save(file=high_distance_path, arr=D_high)

    for experiment, embedding in layouts_dict.items():
        low_distance_path = os.path.join(results_path, "low_distances", experiment)
        os.makedirs(low_distance_path, exist_ok=True)

        if os.path.isfile(low_distance_path + ".npy"):
            D_low = np.load(low_distance_path + ".npy")
        else:
            D_low = compute_distance_list(embedding, eval_distance_metric='euclidean')
            np.save(file=low_distance_path, arr=D_low)
        T, C, R, S, N, CA, SI, DB, SDBW, DSC = compute_all_metrics(x, embedding, D_high, D_low, y,
                                                                    eval_distance_metric=eval_distance_metric)
        results.append([experiment, T, C, R, S, N, CA, SI, DB, SDBW, DSC, str(dict_of_hyperparameter_dicts)])

    new_df = pd.DataFrame(results, columns=["Experiment", "Trustworthiness", "Continuity", "Shephard Diagram "
                                                                                           "Correlation", "Normalized "
                                                                                                          "Stress",
                                            "7-Neighborhood Hit", "Calinski-Harabasz-Index", "Silhouette "
                                                                                             "coefficient",
                                            "Davies-Bouldin-Index", "SDBW validity index", "Distance consistency",
                                            "Complete List of Hyperparameters"])

    if old_df is not None:
        new_df = pd.concat([new_df, old_df])
    new_df.to_csv(res_file_name.replace(".csv", "_partial.csv"), index=False)

    return new_df
