import os
import shutil

import numpy as np
import pandas as pd

from gensim.models import LsiModel

from create_topic_layout import infer_paths_from_base_paths, convert_text_to_corpus
from main import get_raw_dataset
from nlp_standard_preprocessing import load_dataset_if_able


def reformat_top_words_totable_format(selected_results_path, max_columns=100):
    for file in os.listdir(selected_results_path):
        if not file.endswith(".txt") or "top_10_words" not in file or "formatted" in file:
            continue

        file_path = os.path.join(selected_results_path, file)
        num_lines = sum(1 for line in open(file_path))
        num_lines = min(num_lines, max_columns)
        column_header = ("Y|" * num_lines)[:-1]
        new_file_path = file_path.replace("top_10_words", "top_10_words_formatted")
        csv_file_path = file_path.replace("top_10_words.txt", "top_10_words_formatted.csv")
        with open(file_path, "r") as in_file:
            with open(new_file_path, "w+") as out_file:
                with open(csv_file_path, "w+") as csv_file:
                    out_file.write("\\begin{table}[t]\n")
                    out_file.write("        %\\tiny\n")
                    out_file.write("        \\footnotesize\n")
                    out_file.write("        \\setlength{\\tabcolsep}{2.0pt}%\n")
                    out_file.write("        \\renewcommand{\\arraystretch}{1.0}\n")
                    out_file.write("        \\caption{}\n")
                    out_file.write("        \\begin{tabularx}{1.0\\columnwidth}{" + column_header + "}\n")
                    out_file.write("        \\toprule\n")
                    header_line = ""
                    for i in range(num_lines):
                        header_line += "\\textbf{" + str(i) + "} & "
                    header_line = header_line[:-3]
                    header_line += "\\\\ \\midrule"
                    header_line = "        " + header_line + "\n"
                    out_file.write(header_line)

                    line_parts = []
                    counter = 0
                    for line in in_file:
                        line = line.split(",")[1]
                        line = line.replace("(", "").replace(")", "").replace("\n", "")
                        parts = line.split("+")
                        parts = [part.split("*")[1].replace("\"", "").replace("\'", "") for part in parts]
                        line_parts.append(parts)

                        counter += 1
                        if counter == num_lines:
                            break

                    csv_header = ",".join([str(i) for i in range(num_lines)])
                    csv_file.write(csv_header + "\n")

                    for i in range(len(line_parts[0])):
                        words = [part[i] for part in line_parts]
                        csv_line = ",".join([word.strip() for word in words])
                        csv_line += "\n"
                        csv_file.write(csv_line)
                        line = " & ".join(words)
                        line = "        " + line + " \\\\\n"
                        out_file.write(line)

                    out_file.write("        \\hline\n")
                    out_file.write("        \\end{tabularx}\n")
                    out_file.write("        \\label{tab:" + file.split("_")[0] + "}\n")
                    out_file.write("\\end{table}")


def add_alpha_beta_scores_to_results_file(data_base_path, max_davies_bouldin_value, max_calinski_harabasz_value,
                                          required_file_part="collected_metrics", print_to_txt_file=False):
    for cur_dir, dirs, files in os.walk(data_base_path):
        for file in files:
            if not file.endswith(".csv") or required_file_part not in file:
                continue

            if print_to_txt_file:
                target_path = os.path.join(cur_dir, file.replace(".csv", ".txt"))
            else:
                target_path = os.path.join(cur_dir, file)

            df = pd.read_csv(os.path.join(cur_dir, file))
            columns = df.columns.to_list()
            if "alpha" in columns and "beta" in columns:
                return

            seven_neighborhood_index = columns.index("7-Neighborhood Hit")
            trustworthiness_index = columns.index("Trustworthiness")
            continuity_index = columns.index("Continuity")
            sdr_index = columns.index("Shephard Diagram Correlation")
            db_index = columns.index("Davies-Bouldin-Index")
            ch_index = columns.index("Calinski-Harabasz-Index")
            silhou_index = columns.index("Silhouette coefficient")
            distance_consistency_index = columns.index("Distance consistency")

            df["alpha"] = df.apply(
                lambda row: 0.5 * row[seven_neighborhood_index] + 0.5 * ((row[trustworthiness_index] +
                                                                          row[continuity_index] +
                                                                          0.5 * (row[sdr_index]
                                                                                 + 1)) / 3), axis=1)
            df["beta"] = df.apply(
                lambda row: (1 / 3) * (1 - (row[db_index] / max_davies_bouldin_value)) +
                            (1 / 3) * (row[ch_index] / max_calinski_harabasz_value) +
                            (1 / 3) * ((0.5 * (row[silhou_index] + 1) + row[distance_consistency_index]) / 2), axis=1)

            df.to_csv(target_path, index=False)


def get_normalizing_values_from_results_file(dataset_path):
    df = pd.read_csv(dataset_path)
    return df['Davies-Bouldin-Index'].max(), df['Calinski-Harabasz-Index'].max()


def print_line_with_max_alpha_beta(dataset_path, required_file_part):
    for file in os.listdir(dataset_path):
        if not file.endswith(".csv") or required_file_part not in file:
            continue

        df = pd.read_csv(os.path.join(dataset_path, file))
        if "alpha" not in df.columns or "beta" not in df.columns:
            print("Either haven't found alpha or beta for file " + file)
            return

        df = df.sort_values(by="alpha", ascending=False)
        print("Optimal row for alpha:\n" + str(df.iloc[0]) + "\nfor directory " + dataset_path)

        df = df.sort_values(by="beta", ascending=False)
        print("Optimal row for beta:\n" + str(df.iloc[0]) + "\nfor directory " + dataset_path)


def main():
    selected_results_path = "selected_results"
    results_file_path = os.path.join("res_files_only", "results")
    os.makedirs(selected_results_path, exist_ok=True)
    decay = 1.0
    onepass = True
    power_iters = 2
    extra_samples = 100
    models_base = "models"
    results_base = "results"
    eval_datasets = ["20_newsgroups", "emails", "reuters", "seven_categories", "github_projects"]

    for dataset in eval_datasets:
        if "statistical_analysis" in dataset:
            continue

        dataset_path = os.path.join(results_file_path, dataset)
        max_davies_bouldin, max_calinski_harabasz \
            = get_normalizing_values_from_results_file(os.path.join(dataset_path,
                                                                    "full_res_" + dataset + ".csv"))
        add_alpha_beta_scores_to_results_file(dataset_path,
                                              max_davies_bouldin_value=max_davies_bouldin,
                                              max_calinski_harabasz_value=max_calinski_harabasz,
                                              required_file_part="full_res")
        print_line_with_max_alpha_beta(dataset_path, required_file_part="full_res")
        create_csv_from_npy_file(dataset_name=dataset, selected_res_path="optimal_results")

    for dataset_name in eval_datasets:
        get_selected_results(dataset_name, results_base, selected_results_path)
        get_lsi_top_10_words(dataset_name, decay, extra_samples, models_base, onepass, power_iters,
                             selected_results_path)

    reformat_top_words_totable_format(selected_results_path)


def create_csv_from_npy_file(dataset_name, selected_res_path):
    _, _, y = get_true_x_y(dataset_name)
    y = y.astype(int)

    for file in os.listdir(selected_res_path):
        if not file.endswith(".npy") or dataset_name not in file:
            continue

        file_path = os.path.join(selected_res_path, file)
        layout_data = np.load(file_path)
        concatenated = np.vstack((layout_data.T, y)).T
        df = pd.DataFrame(concatenated, columns=['x', 'y', 'category'])
        df.to_csv(file_path.replace(".npy", ".csv"), index=False)


def get_selected_results(dataset_name, results_base, selected_results_path):
    res_path = os.path.join(results_base, dataset_name)
    selected_res_path = os.path.join(selected_results_path, res_path)
    os.makedirs(selected_res_path, exist_ok=True)

    if not os.path.isdir(res_path):
        create_csv_from_npy_file(dataset_name, selected_res_path)
        return

    for file in os.listdir(res_path):
        file = str(file)
        if 'tsne' in file and 'auto' in file and 'lsi' in file and 'tfidf' in file and file.endswith('.npy'):
            shutil.copy(os.path.join(res_path, file), os.path.join(selected_res_path, file))
        elif 'tsne' in file and dataset_name + "_tfidf" in file and 'auto' in file and file.endswith('.npy'):
            shutil.copy(os.path.join(res_path, file), os.path.join(selected_res_path, file))

    create_csv_from_npy_file(dataset_name, selected_res_path)


def get_lsi_top_10_words(dataset_name, decay, extra_samples, models_base, onepass, power_iters, selected_results_path):
    dest_file = os.path.join(selected_results_path, dataset_name + "_lsi_tfidf_top_10_words.txt")
    special_topics = {'ecommerce': 8, 'seven_categories': 14, 'emails': 8, 'github_projects': 16}

    min_density, x, y = get_true_x_y(dataset_name)
    dictionary, corpus = convert_text_to_corpus(x)

    if dataset_name in special_topics.keys():
        n_topics = special_topics[dataset_name]
    else:
        n_topics = len(np.unique(y))
    model_base_path = os.path.join(models_base, dataset_name)
    base_path_lsi = os.path.join(model_base_path,
                                 "lsi_" + str(n_topics) + "_" + str(decay) + "_" + str(onepass) + "_" + str(
                                     power_iters) + "_" + str(extra_samples) + "_" + str(len(dictionary)))
    base_path_tfidf = base_path_lsi.replace("lsi", "lsi_tfidf")
    model_path_tfidf, dense_matrix_path_tfidf, linear_matrix_path_tfidf = infer_paths_from_base_paths(
        base_path_tfidf)
    if os.path.isfile(model_path_tfidf):
        model = LsiModel.load(model_path_tfidf)
    else:
        tfidf_path = os.path.join(model_base_path, "tfidf_model_" + str(min_density) + "_" + str(len(dictionary)))
        tfidf_path_sparse = tfidf_path.replace("tfidf_model", "tfidf_model_sparse")
        tfidf_sparse = np.load(tfidf_path_sparse + ".npy", allow_pickle=True)
        model = LsiModel(tfidf_sparse, id2word=dictionary, num_topics=n_topics, decay=decay, onepass=onepass,
                         power_iters=power_iters, extra_samples=extra_samples, random_seed=0)
        model.save(model_path_tfidf)
    lines = model.show_topics(num_topics=-1, num_words=10)
    with open(dest_file, "w+") as out_file:
        for line in lines:
            out_file.write(f"{line}\n")


def get_true_x_y(dataset_name):
    min_density, _, _, x, y = get_raw_dataset(dataset_name)
    dataset_dir = os.path.join("data", dataset_name)
    file_path = os.path.join(dataset_dir, dataset_name + "_words_list_" + str(len(x)) + ".pkl")
    print("Try to load dataset from: " + file_path, flush=True)
    x = load_dataset_if_able(file_path)
    to_discard = [i for i, text in enumerate(x) if len(text) <= 1]
    x = [x[i] for i in range(len(x)) if i not in to_discard]
    y = np.array([y[i] for i in range(len(y)) if i not in to_discard])
    return min_density, x, y


if __name__ == "__main__":
    main()
