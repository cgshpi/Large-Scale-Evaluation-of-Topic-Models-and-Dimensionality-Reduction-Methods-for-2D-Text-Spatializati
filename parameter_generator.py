from argparse import ArgumentParser, ArgumentTypeError
import itertools
import os
import shutil
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = ArgumentParser(description="This script generates a csv with all call strings and res_files")
    parser.add_argument('--n_lines', dest='n_lines', type=int, default=10000,
                        help="Specifies the number of parameter configurations generated. Defaults to 10000")
    parser.add_argument('--res_dir_path', dest='res_path', type=str, default="./",
                        help="If the cwd of the script differs from the location where the call to the main.py"
                             " script is invoked, "
                             "we have to search for the file in another directory. This parameter shall point "
                             "to this location (cwd in context of the main.py script. Defaults to './'")
    parser.add_argument('--only_som', dest='only_som', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the SOM shall be processed or all other projection techniques.")
    parser.add_argument('--only_tsne', dest='only_tsne', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the t-SNE shall be processed or all other projection techniques.")
    parser.add_argument('--get_all', dest='get_all', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the SOM shall be processed or all other projection techniques.")

    parser.add_argument('--disable_model_training', dest='dmt', type=str2bool, default=False, const=True, nargs="?",
                        help="If set the program will return after preprocessing the requested dataset and "
                             "creating the topic models. No evaluation is undertaken in this case.")
    parser.add_argument('--request_default', dest='rd', type=str2bool, default=False, const=True, nargs="?",
                        help="If set only the default parameters of a DR will be evaluated")

    parser.add_argument('--big_datasets', dest='bd', type=str2bool, default=False, const=True, nargs="?",
                        help="If set, get parameters for big datasets")
    args = parser.parse_args()
    return args.n_lines, args.res_path, args.only_som, args.get_all, args.dmt, args.bd, args.only_tsne, args.rd


def main():
    lines, res_dir_path, only_som, get_all_parameters, disable_model_training, request_big_datasets, only_tsne,\
        request_default = get_arguments()

    if request_big_datasets:
        available_datasets = ["github_projects"]
    else:
        available_datasets = [
            "20_newsgroups",
            "reuters",
            "emails",
            "seven_categories"
        ]

    available_tms = [
        "bow",
        "tfidf",
        "lda",
        "lda_linear_combined",
        "lsi",
        "lsi_linear_combined",
        "lsi_tfidf",
        "lsi_tfidf_linear_combined",
        "nmf",
        "nmf_linear_combined",
        "nmf_tfidf",
        "nmf_tfidf_linear_combined",
        "bert"
    ]
    special_topics = {'ecommerce': 8, 'seven_categories': 14, 'emails': 8, 'github_projects': 16}
    perplexity_tsne = [5 * (i + 1) for i in range(0, 10)]
    learning_rate_tsne = [10, 17, 28, 46, 77, 129, 215, 359, 599, 1000]  # Logarithmic scale
    n_iter_tsne = [250, 1000, 2000, 4000, 10000]

    n_neighbors_umap = [10 * (i + 1) for i in range(0, 10)]
    n_neighbors_umap.append(2)
    min_dist_umap = [i / 10.0 for i in range(0, 11)]
    runs_mds = [300 + i * 20 for i in range(0, 30)]
    n_som = [i for i in range(10, 20)]
    m_som = [i for i in range(10, 20)]
    full_list_tsne = list(itertools.product(perplexity_tsne, n_iter_tsne, learning_rate_tsne))
    full_list_umap = list(itertools.product(n_neighbors_umap, min_dist_umap))
    full_list_som = list(itertools.product(n_som, m_som))

    if request_default:
        full_list_tsne = [(30, 1000, 'auto')]
        full_list_umap = [(15, 0.1)]
        runs_mds = [300]

    if only_som:
        parameter_list = get_som_parameter_list(full_list_som)
    elif only_tsne:
        parameter_list = get_tsne_parameter_list(full_list_tsne)
    else:
        max_length = max(len(full_list_tsne), len(full_list_umap), len(runs_mds))
        parameter_list = ["main.py"] * max_length

        for i in range(len(full_list_tsne)):
            parameter_list[i] += " --perplexity_tsne " + str(full_list_tsne[i][0]) + " --n_iter_tsne " + \
                                 str(full_list_tsne[i][1]) + " --learning_rate " + str(full_list_tsne[i][2])

        for i in range(len(full_list_umap)):
            parameter_list[i] += " --n_neighbors_umap " + str(full_list_umap[i][0]) + " --min_dist_umap " + \
                                 str(full_list_umap[i][1])

        for i in range(len(runs_mds)):
            parameter_list[i] += " --max_iter_mds " + str(runs_mds[i])

        if get_all_parameters:
            parameter_list.extend(get_som_parameter_list(full_list_som))
            random.seed(42)
            random.shuffle(parameter_list)

    parameter_lists_tmp = []
    # print(((len(perplexity_tsne) * len(learning_rate_tsne) * len(n_iter_tsne)) + (len(n_neighbors_umap) * len(min_dist_umap))
    #       + (len(n_som) * len(m_som)) + len(runs_mds) + 1) * len(available_tms) * len(available_datasets))
    for dataset in available_datasets:
        if dataset in special_topics.keys():
            parameter_lists_tmp.extend([parameters + " --dataset_name " + dataset + " --n_topics_lda "
                                        + str(special_topics[dataset]) + " --n_topics_lsi " +
                                        str(special_topics[dataset]) + " --n_topics_nmf " +
                                        str(special_topics[dataset]) for parameters in parameter_list])
        else:
            parameter_lists_tmp.extend([parameters + " --dataset_name " + dataset for parameters in parameter_list])

    if disable_model_training:
        disable_model_parameter_string_part = " --disable_model_training"
    else:
        disable_model_parameter_string_part = ""
    parameter_lists_tmp = [el + disable_model_parameter_string_part for el in parameter_lists_tmp]
    parameter_list = parameter_lists_tmp

    parameter_lists_tmp = []
    for available_tm in available_tms:
        parameter_lists_tmp.extend([parameter_string + " --topic_model " + available_tm for parameter_string
                                    in parameter_list])
    parameter_list = parameter_lists_tmp
    # print(len(parameter_list))

    for i, parameter_string in enumerate(parameter_list):
        dataset = parameter_string.split("dataset_name")[1].strip().split(" ")[0]

        file_name = parameter_string.replace("main.py ", "").replace("-", "")
        if dataset in special_topics.keys():
            file_name = file_name.replace("n_topics_lda " + str(special_topics[dataset]), "") \
                .replace("n_topics_lsi " + str(special_topics[dataset]), "").replace("_nmf", "")
        if disable_model_training:
            file_name = file_name.replace("disable_model_training", "")
        file_name = ' '.join(file_name.split())
        file_name = file_name.replace(" ", "_")
        file_name = '_'.join(file_name.split("_"))

        file_name = str(os.path.join("results", dataset, "results_"
                                     + file_name + ".csv"))
        if not os.path.isfile("." + os.path.sep + file_name) or get_all_parameters:
            print(res_dir_path + file_name + "," + parameter_string + " --res_file_name " + "." + os.path.sep + file_name)
        else:
            for dataset_name in available_datasets:
                os.makedirs(os.path.join("res_files_only", "results", dataset_name), exist_ok=True)
            shutil.copyfile(("." + os.path.sep + file_name), os.path.join("res_files_only", file_name))


def get_som_parameter_list(full_list_som):
    parameter_list = ["main.py"] * len(full_list_som)
    for i in range(len(full_list_som)):
        parameter_list[i] += " --n_som " + str(full_list_som[i][0]) + " --m_som " + str(full_list_som[i][1]) \
                             + " --only_som=True"
    return parameter_list


def get_tsne_parameter_list(full_list_tsne):
    parameter_list = ["main.py"] * len(full_list_tsne)
    for i in range(len(full_list_tsne)):
        parameter_list[i] += " --perplexity_tsne " + str(full_list_tsne[i][0]) + " --n_iter_tsne " + \
                             str(full_list_tsne[i][1]) + " --learning_rate " + str(full_list_tsne[i][2]) +\
                             " --only_tsne "
    return parameter_list


if __name__ == "__main__":
    main()
