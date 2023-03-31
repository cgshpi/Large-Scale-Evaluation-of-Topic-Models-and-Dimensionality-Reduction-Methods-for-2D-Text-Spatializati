import os
import ast
from argparse import ArgumentParser

import pandas as pd
import numpy as np

datasets = ["20_newsgroups", "reuters", "github_projects", "seven_categories", "emails", "ecommerce"]
available_dimension_reductions = ["som", "umap", "tsne", "mds"]
available_parametrized_topic_models = ["lda", "lsi", "bow", "tfidf", "lsi_tfidf", "nmf", "nmf_tfidf", "bert"]
parametrized_topic_models = ["lda", "lsi", "lsi_tfidf"]


def get_arguments():
    parser = ArgumentParser(description="This script generates a csv with all call strings and res_files")
    parser.add_argument('--res_dir_path', dest='res_path', type=str, default=os.path.join("res_files_only", "results"),
                        help="A root directory containing a directory for each dataset where the results files for"
                             "this dataset reside in.")
    args = parser.parse_args()
    return args.res_path


def find_parts_to_exclude(parameter_list, dr="mds", attribute="max_iter", min_value=300, max_value=900):
    truth_values = []
    n_iters = set()
    for el in parameter_list:
        if type(el) == str:
            el = ast.literal_eval(el)

        if dr not in el.keys():
            truth_values.append(True)
        else:
            n_iter = el[dr][attribute]
            n_iters.add(n_iter)
            if n_iter < min_value or n_iter > max_value:
                truth_values.append(False)
            else:
                truth_values.append(True)

    print(n_iters)
    return truth_values


def exclude_old_results(res_dirs_base_path):
    for dataset in datasets:
        dataset_path = os.path.join(res_dirs_base_path, dataset)
        full_res_file_path = os.path.join(dataset_path, "full_res_" + dataset + ".csv")
        df = pd.read_csv(full_res_file_path)
        df = df.loc[find_parts_to_exclude(df["Complete List of Hyperparameters"].to_numpy())]
        print(len(df))
        df.to_csv(full_res_file_path, index=False)


def main():
    res_dirs_base_path = get_arguments()

    # exclude_old_results(res_dirs_base_path)
    for dataset in datasets:
        res_df = None
        cur_path = os.path.join(res_dirs_base_path, dataset)
        for file in os.listdir(cur_path):
            if file.endswith(".csv") and "full" not in file:
                file_path = os.path.join(cur_path, file)
                parts = file.split("_")
                cur_drs = [dr for dr in available_dimension_reductions if dr in parts]
                cur_df = pd.read_csv(file_path)
                experiment_names = cur_df["Experiment"].to_numpy()
                cur_indices = [i for i, experiment_name in enumerate(experiment_names) if
                               any([cur_dr in experiment_name for cur_dr in cur_drs])]
                cur_df = cur_df.iloc[cur_indices]
                experiment_names = cur_df["Experiment"].to_numpy()
                parameters = cur_df["Complete List of Hyperparameters"].to_numpy()
                run_parameters = []
                run_tms = []
                run_drs = []
                for experiment_name, parameter_list in zip(experiment_names, parameters):
                    parts = experiment_name.split("_")
                    if "lsi" in parts and "tfidf" in parts:
                        experiment_tm = "lsi_tfidf"
                    elif "nmf" in parts and "tfidf" in parts:
                        experiment_tm = "nmf_tfidf"
                    else:
                        experiment_tm = [tm for tm in available_parametrized_topic_models if tm in parts][0]
                    experiment_dr = [dr for dr in available_dimension_reductions if dr in parts][0]

                    parameter_list = ast.literal_eval(parameter_list)
                    if "combined" in parts:
                        run_tms.append(experiment_tm + "_linear_combined")
                    else:
                        run_tms.append(experiment_tm)
                    run_drs.append(experiment_dr)
                    if experiment_tm in parametrized_topic_models:
                        if experiment_tm == "lsi_tfidf":
                            run_parameters.append({experiment_tm: parameter_list["lsi"],
                                                   experiment_dr: parameter_list[experiment_dr]})
                        else:
                            run_parameters.append({experiment_tm: parameter_list[experiment_tm],
                                                   experiment_dr: parameter_list[experiment_dr]})
                    else:
                        run_parameters.append({experiment_dr: parameter_list[experiment_dr]})

                cur_df["Complete List of Hyperparameters"] = np.array([str(run_parameter) for run_parameter
                                                                       in run_parameters])
                cur_df["DR"] = np.array(run_drs)
                cur_df["TM"] = np.array(run_tms)
                if res_df is None:
                    res_df = cur_df
                else:
                    res_df = pd.concat([res_df, cur_df])
            else:
                continue
        if res_df is None:
            continue

        res_df = res_df.loc[res_df["Normalized Stress"].notnull()]  # Sanity check, that lines are captured entirely
        res_df = res_df.drop_duplicates(subset=["Experiment", "TM", "DR", "Complete List of Hyperparameters"],
                                        keep="last")
        print(dataset + " " + str(len(res_df)))
        res_df = res_df.sort_values(by=["Experiment", "TM", "DR"])
        res_df.to_csv(os.path.join(cur_path, "full_res_" + dataset + ".csv"), index=False)


if __name__ == "__main__":
    main()
