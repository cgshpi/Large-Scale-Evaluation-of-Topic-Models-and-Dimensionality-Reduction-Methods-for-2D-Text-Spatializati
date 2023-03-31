import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import ast

import scipy.stats as stats
from scipy.stats import ConstantInputWarning
from statsmodels.stats.descriptivestats import sign_test
import matplotlib.pyplot as plt
import warnings


def get_arguments():
    parser = ArgumentParser(description="This script generates a csv with all call strings and res_files")
    parser.add_argument('--res_dir_path', dest='res_path', type=str, default=os.path.join("res_files_only", "results"),
                        help="A root directory containing a directory for each dataset where the results files for"
                             "this dataset reside in.")
    args = parser.parse_args()
    return args.res_path


def normalize_input(X):
    if np.max(X) > 1.0 or np.min(X) < 0.0:
        return (X - np.min(X)) / (np.max(X) - np.min(X))
    else:
        return X


def is_constant_row(column, sub_df):
    values = sub_df[column].to_numpy()
    return np.all(np.isclose(values, values[0]))


def perform_statistical_tests(result_file, project, result_dir="res_files_only"):
    espadoto_metrics = ["Trustworthiness", "Continuity", "Shephard Diagram Correlation", "Normalized Stress",
                        "7-Neighborhood Hit"]
    cluster_metrics = ["Calinski-Harabasz-Index", "Silhouette coefficient", "Davies-Bouldin-Index",
                       "SDBW validity index"]
    all_metrics = espadoto_metrics + cluster_metrics
    os.makedirs(result_dir, exist_ok=True)

    results_df = pd.read_csv(result_file)
    dimension_reduction_list = results_df["DR"].to_numpy()
    topic_model_list = results_df["TM"].to_numpy()
    dimension_reductions = np.unique(dimension_reduction_list)
    topic_models = np.unique(topic_model_list)

    for dr, tm in zip(dimension_reductions, topic_models):
        parameters = dict()
        indices = [i for i in range(len(results_df)) if dimension_reduction_list[i] == dr and topic_model_list[i] == tm]
        sub_df = results_df.iloc[indices]
        list_of_hyperparameters = sub_df["Complete List of Hyperparameters"].to_list()
        list_of_hyperparameters = [ast.literal_eval(hyperparameter_dict) for hyperparameter_dict in
                                   list_of_hyperparameters]

        hyperparameters_dr = list_of_hyperparameters[0][dr].keys()
        for hyperparameter in hyperparameters_dr:
            parameters[hyperparameter] = [hyperparameter_dict[dr][hyperparameter] for hyperparameter_dict in
                                          list_of_hyperparameters]

        if tm in list_of_hyperparameters[0].keys():
            hyperparameters_tm = list_of_hyperparameters[0][tm].keys()
            for hyperparameter in hyperparameters_tm:
                parameters[hyperparameter] = [hyperparameter_dict[tm][hyperparameter] for hyperparameter_dict in
                                              list_of_hyperparameters]

        sub_df.drop(columns=["Complete List of Hyperparameters", "DR", "TM", "Experiment"], inplace=True)
        for hyperparameter, values in parameters.items():
            sub_df[hyperparameter] = pd.Series(values)
        parameter_columns = list(parameters.keys())
        sub_df = sub_df.fillna(0.0)
        all_metrics_non_constant = [metrics for metrics in all_metrics if not is_constant_row(metrics, sub_df)]

        if len(all_metrics_non_constant) == 0:
            continue

        espadoto_metrics_non_constant = [metrics for metrics in espadoto_metrics
                                         if not is_constant_row(metrics, sub_df)]
        cluster_metrics_non_constant = [metrics for metrics in cluster_metrics if not is_constant_row(metrics, sub_df)]
        cur_res_file = result_file + "_" + dr + "_" + tm

        evaluate_results(all_metrics_non_constant, cluster_metrics_non_constant, espadoto_metrics_non_constant,
                         parameter_columns, project, result_dir, cur_res_file, sub_df)


def evaluate_results(all_metrics, cluster_metrics, espadoto_metrics, parameters_columns, project,
                     result_dir, result_file, results_df, min_null=None):
    if min_null is None:
        min_null = [0.1, 0.05, 0.01, 0.001]
    top_rows = []
    for metric in all_metrics:
        metric_values = results_df[metric].to_numpy()
        if metric in cluster_metrics:
            metric_values = np.absolute(metric_values)
        metric_values = normalize_input(metric_values)
        if metric in cluster_metrics:
            metric_values = 1. - metric_values
        max_pos = np.argpartition(metric_values, -5)[-5:]
        max_pos = max_pos[np.argsort(metric_values[max_pos])]
        max_row = results_df.iloc[max_pos].to_numpy()
        max_row = max_row.tolist()
        for i, row in enumerate(max_row):
            row.append("Max " + str(5 - i) + " for " + project + " and metric " + metric)
            top_rows.append(row)

    summed_metrics = None
    for metric in espadoto_metrics:
        metric_values = results_df[metric].to_numpy()
        metric_values = normalize_input(metric_values)
        if summed_metrics is not None:
            summed_metrics += metric_values
        else:
            summed_metrics = metric_values
    max_pos = np.argpartition(summed_metrics, -5)[-5:]
    max_pos = max_pos[np.argsort(summed_metrics[max_pos])]
    max_row = results_df.iloc[max_pos].to_numpy()
    max_row = max_row.tolist()
    for i, row in enumerate(max_row):
        row.append("Max " + str(5 - i) + " for " + project + " and sum of espadoto metrics")
        top_rows.append(row)

    summed_metrics_2 = None
    for metric in cluster_metrics:
        metric_values = results_df[metric].to_numpy()
        metric_values = np.absolute(metric_values)
        metric_values = normalize_input(metric_values)
        metric_values = 1. - metric_values
        if summed_metrics_2 is not None:
            summed_metrics_2 += metric_values
        else:
            summed_metrics_2 = metric_values
    max_pos = np.argpartition(summed_metrics_2, -5)[-5:]
    max_pos = max_pos[np.argsort(summed_metrics_2[max_pos])]
    max_row = results_df.iloc[max_pos].to_numpy()
    max_row = max_row.tolist()
    for i, row in enumerate(max_row):
        row.append("Max " + str(5 - i) + " for " + project + " and sum of cluster metrics")
        top_rows.append(row)

    summed_total = summed_metrics + summed_metrics_2
    max_pos = np.argpartition(summed_total, -5)[-5:]
    max_pos = max_pos[np.argsort(summed_total[max_pos])]
    max_row = results_df.iloc[max_pos].to_numpy()
    max_row = max_row.tolist()
    for i, row in enumerate(max_row):
        row.append("Max " + str(5 - i) + " for " + project + " and sum of all metrics")
        top_rows.append(row)
    columns = results_df.columns.tolist()
    columns.append("Criterion")
    max_df = pd.DataFrame(data=np.array(top_rows), columns=columns)
    max_df.to_csv(os.path.join(result_dir, "evaluation_" + result_file.split(os.sep)[-1]))
    statistic_test_values = []
    for metric in all_metrics:
        metric_values = results_df[metric].to_numpy()
        if metric in cluster_metrics:
            metric_values = np.nan_to_num(metric_values, nan=max(abs(np.min(metric_values)), np.max(metric_values)))
        else:
            metric_values = np.nan_to_num(metric_values)
        if metric in cluster_metrics:
            metric_values = np.absolute(metric_values)
            metric_values = normalize_input(metric_values)
            metric_values = 1. - metric_values
        else:
            metric_values = normalize_input(metric_values)
        stats.probplot(metric_values, dist="norm", plot=plt)
        plt.title("Probability Plot - " + metric)
        plt.savefig(os.path.join(result_dir, metric + "_q_q_plot.png"))
        plt.close()
        for parameter in parameters_columns:
            parameter_values = results_df[parameter].to_numpy()
            try:
                float(parameter_values[0])
            except ValueError:
                unique = np.unique(parameter_values)
                mapping = {value: i for i, value in enumerate(unique)}
                parameter_values = [mapping[value] for value in parameter_values]
            nan_indices = np.argwhere(np.isnan(parameter_values))
            metric_values = [metric_value for i, metric_value in enumerate(metric_values) if i not in nan_indices]
            parameter_values = [parametric_value for i, parametric_value in enumerate(parameter_values)
                                if i not in nan_indices]
            parameter_values = np.absolute(parameter_values)
            parameter_values = normalize_input(parameter_values)
            parameter_values = parameter_values - np.mean(parameter_values)
            metric_values = metric_values - np.mean(metric_values)
            try:
                differences = parameter_values - metric_values
            except:
                continue
            # Pearson would assume normal distribution
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    spearman_statistic = stats.spearmanr(a=metric_values, b=parameter_values)
                except ConstantInputWarning:
                    # If the input is constant then there is no point in analysing this any further
                    print("Handled constant input")
                    continue
            statistic_test_values.append([metric, parameter,
                                          stats.wilcoxon(x=metric_values, y=parameter_values).pvalue,
                                          stats.mannwhitneyu(x=metric_values, y=parameter_values).pvalue,
                                          sign_test(differences)[1],
                                          spearman_statistic[0], spearman_statistic[1]
                                          ])

    statistic_test_values = pd.DataFrame(data=statistic_test_values,
                                         columns=["Metric", "Parameter", "Wilcoxon_statistic_p_value",
                                                  "Mann_Whitney_u_test_p_value", "Sign_test_for_differences_p_value",
                                                  "Spearman_correlation_statistic", "Spearman_correlation_p_value"])
    statistic_test_values.to_csv(os.path.join(result_dir, "statistical_" + result_file.split(os.sep)[-1]))

    spearman_p_values = statistic_test_values["Spearman_correlation_p_value"].to_numpy()
    wilcoxon_values = statistic_test_values["Wilcoxon_statistic_p_value"].to_numpy()
    mann_values = statistic_test_values["Mann_Whitney_u_test_p_value"].to_numpy()
    sign_values = statistic_test_values["Sign_test_for_differences_p_value"].to_numpy()

    for cut_off_value in min_null:
        over = [i for i, value in enumerate(spearman_p_values) if value < cut_off_value],
        null_df = statistic_test_values.iloc[over]
        null_df.to_csv(os.path.join(result_dir, "correlation_null_hypothesis_unlikely_" + str(cut_off_value) + "_" +
                                    result_file.split(os.sep)[-1]))

        over = np.intersect1d(over, [i for i, value in enumerate(wilcoxon_values) if value > cut_off_value])
        over = np.intersect1d(over, [i for i, value in enumerate(mann_values) if value > cut_off_value])
        over = np.intersect1d(over, [i for i, value in enumerate(sign_values) if value > cut_off_value])

        null_df = statistic_test_values.iloc[over]
        null_df.to_csv(os.path.join(result_dir, "same_distribution_uphold_correlation_likely_" + str(cut_off_value)
                                    + "_" + result_file.split(os.sep)[-1]))


def main():
    res_dir_path = get_arguments()
    for root, dirs, files in os.walk(res_dir_path):
        for file in files:
            if "full_res" in file and "statistical_analysis" not in root:
                file_path = os.path.join(root, file)
                project = root.split(os.path.sep)[-1]
                perform_statistical_tests(file_path, project, result_dir=root + "_statistical_analysis")


if __name__ == "__main__":
    main()