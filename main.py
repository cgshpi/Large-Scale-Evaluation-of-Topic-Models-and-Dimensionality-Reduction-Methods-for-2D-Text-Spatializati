#!/bin/env python3
import csv
import gc
import pickle
import shutil
import time

import pandas as pd
import psutil
import humanize
from sklearn.model_selection import train_test_split

print("Started script", flush=True)
from nltk_download import download_nltk_resources
from nltk.corpus import reuters

download_nltk_resources()
from sklearn.datasets import fetch_20newsgroups
from argparse import ArgumentParser, ArgumentTypeError
import os
import numpy as np

from create_topic_layout import create_all_topic_models
from metrics import evaluate_layouts
from projections import create_layouts
from nlp_standard_preprocessing import preprocess_texts, load_dataset_if_able


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_float_value(v, accepted_strings=None, argument_name=""):
    if accepted_strings is None:
        accepted_strings = []
    try:
        v = float(v)
        return v
    except ValueError:
        if v in accepted_strings:
            return v
        else:
            raise ArgumentTypeError('Argument ' + argument_name + " has either to be float or in " +
                                    str(accepted_strings))


def get_arguments():
    parser = ArgumentParser(description="These scripts create all layouts for all datasets for one hyperparameter"
                                        "configuration. Please note, that you will handle the hyperparameter loop"
                                        "from outside. Compare to sbatch.")
    parser.add_argument('--n_topics_lda', dest='n_lda', type=int, default=None,
                        help="Specifies the number of topics for MalletLDA. Defaults to number of categories")
    parser.add_argument('--alpha_lda', dest='alpha_lda', type=str, default='auto',
                        help='Specifies the alpha hyperparameter for LDA. Defaults to auto')
    parser.add_argument('--n_iterations_lda', dest='iterations_lda', type=int, default=1000,
                        help='Specifies the number of iterations MalletLDA will take. A too high number may lead'
                             'to overfitting, but too few to underfitting. Defaults to 1000')
    parser.add_argument('--topic_threshold_lda', dest='t_lda', type=float, default=0.0,
                        help='Threshold of the probability above which we consider a topic. Defaults to 0.0')

    parser.add_argument('--n_topics_lsi', dest='n_lsi', type=int, default=None,
                        help='Specifies the number of topics of GensimLSI. Defaults to number of categories')
    parser.add_argument('--decay_lsi', dest='decay_lsi', type=float, default=1.0,
                        help="Specifies the decay value for GensimLSI. Defaults to 1.0")
    parser.add_argument('--onepass_lsi', dest='one_lsi', type=str2bool, default=True,
                        help="Specifies whether or not onepass is used for LSI. Defaults to True")
    parser.add_argument("--power_iters_lsi", dest='pi_lsi', type=int, default=2,
                        help="Specifies the power iterations for LSI. Defaults to 2")
    parser.add_argument("--extra_samples_lsi", dest="es_lsi", type=int, default=100,
                        help="Specifies the number of extra samples used by LSI. Defaults to 100")

    parser.add_argument('--n_topics_nmf', dest='n_nmf', type=int, default=None,
                        help='Specifies the number of topics of GensimNMF. Defaults to number of categories')

    parser.add_argument('--n_categories_bert', dest='n_bert', type=int, default=None,
                        help='Specifies the number of categories for BERT. Defaults to number of categories')

    parser.add_argument('--perplexity_tsne', dest='p_tsne', type=float, default=30.0,
                        help='Specifies the perplexity for t-SNE. Defaults to 30.0')
    parser.add_argument('--early_exaggeration_tsne', dest='e_tsne', type=float, default=12.0,
                        help="Specifies the early exaggeration value for GensimLSI. Defaults to 12.0")
    parser.add_argument('--learning_rate', dest='l_tsne', default='auto',
                        help="Has either to be auto or a floating point number. Defaults to auto")
    parser.add_argument("--n_iter_tsne", dest='n_tsne', type=int, default=250,
                        help="Specifies the number of iterations for tsne. Defaults to 250")
    parser.add_argument("--angle_tsne", dest="a_tsne", type=float, default=0.5,
                        help="Specifies the angle for the Barnes-Hut method. Defaults to 0.5")

    parser.add_argument('--n_neighbors_umap', dest='n_umap', type=int, default=15,
                        help='Specifies the number of neighbors considered by UMAP. Defaults to 15')
    parser.add_argument('--min_dist_umap', dest='m_umap', type=float, default=0.1,
                        help="Specifies the minimum distance for UMAP. Defaults to 0.1")
    parser.add_argument('--metric_umap', dest='me_umap', type=str, default='cosine',
                        help="The metric used by UMAP. Defaults to cosine")
    parser.add_argument("--spread_umap", dest='s_umap', type=float, default=1.0,
                        help="Specifies the spread for UMAP. Defaults to 1.0")
    parser.add_argument("--set_op_mix_ratio_umap", dest="somr_umap", type=float, default=1.0,
                        help="Specifies the op_mix ratio. See the documentation of UMAP for more details."
                             " Defaults to 1.0")
    parser.add_argument("--local_connectivity", dest="lc_umap", type=int, default=1,
                        help="Specifies the layer connectivity. See the documentation of UMAP for more details."
                             " Defaults to 1")
    parser.add_argument("--repulsion_strength_umap", dest="rs_umap", type=float, default=1.0,
                        help="Specifies the repulsion strength. See the documentation of UMAP for more details."
                             " Defaults to 1.0")
    parser.add_argument("--negative_sample_rate_umap", dest="nsr_umap", type=int, default=5,
                        help="Specifies the negative sample rate. See the documentation of UMAP for more details."
                             " Defaults to 5")

    parser.add_argument("--max_iter_mds", dest="i_mds", type=int, default=300,
                        help="Specifies the metric used for MDS.")

    parser.add_argument('--n_som', dest='n_som', type=int, default=10,
                        help='The n parameter of SOM. Defaults to 10')
    parser.add_argument('--m_som', dest='m_som', type=int, default=10,
                        help='The m parameter of SOM. Defaults to 10')

    parser.add_argument("--res_file_name", dest="res_file", type=str, default=os.path.join(os.getcwd(), "results.csv"),
                        help="Specifies the file name where results shall be written to.")
    parser.add_argument("--dataset_name", dest="d", type=str, default="20_newsgroups",
                        help="Specifies the used dataset. Defaults to the 20 newsgroups dataset.")
    parser.add_argument('--only_som', dest='only_som', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the SOM shall be processed or all other projection techniques.")
    parser.add_argument('--only_tsne', dest='only_tsne', type=str2bool, default=False, const=True, nargs="?",
                        help="Specifies whether only the t-SNE shall be processed or all other projection techniques.")

    parser.add_argument('--only_precalculate', dest='op', type=str2bool, default=False, const=True, nargs="?",
                        help="If set the program will return after preprocessing the requested dataset and "
                             "creating the topic models. No evaluation is undertaken in this case.")

    parser.add_argument('--disable_model_training', dest='dmt', type=str2bool, default=False, const=True, nargs="?",
                        help="If set the program will return after preprocessing the requested dataset and "
                             "creating the topic models. No evaluation is undertaken in this case.")
    parser.add_argument('--force_reeval', dest='fr', type=str2bool, default=False, const=True, nargs="?",
                        help="If set, a reevaluation will be forced")

    parser.add_argument("--topic_model", dest="tm", type=str, default="", nargs="?",
                        help="If set determines which single topic model will be loaded")

    args = parser.parse_args()

    parameter_dict = {'lda': {'n_topics': args.n_lda, 'alpha': str_or_float_value(
        args.alpha_lda, accepted_strings=["symmetric", "asymmetric", "auto"], argument_name="Alpha LDA"),
                              'iterations': args.iterations_lda,
                              'topic_threshold': args.t_lda},
                      'lsi': {'n_topics': args.n_lsi, 'decay': args.decay_lsi, 'onepass': args.one_lsi,
                              'power_iters': args.pi_lsi, 'extra_samples': args.es_lsi},
                      'nmf': {'n_topics': args.n_nmf},
                      'bert': {'n_categories': args.n_bert},
                      'tsne': {'perplexity': args.p_tsne, 'early_exaggeration': args.e_tsne,
                               'learning_rate': str_or_float_value(args.l_tsne, accepted_strings=["auto"],
                                                                   argument_name="Learning rate t-SNE"),
                               'n_iter': args.n_tsne, 'angle': args.a_tsne},
                      'umap': {'n_neighbors': args.n_umap, 'min_dist': args.m_umap, 'metric': args.me_umap,
                               'spread': args.s_umap, 'set_op_mix_ratio': args.somr_umap,
                               'local_connectivity': args.lc_umap, 'repulsion_strength': args.rs_umap,
                               'negative_sample_rate': args.nsr_umap},
                      'pca': {},
                      'mds': {'max_iter': args.i_mds},
                      'som': {'n': args.n_som, 'm': args.m_som}}

    return parameter_dict, args.res_file, args.d, args.only_som, args.op, args.dmt, args.tm, args.only_tsne, args.fr


def print_current_memory_usage():
    print("Current memory usage: " + str(humanize.naturalsize(psutil.Process(os.getpid()).memory_info().rss)),
          flush=True)


def main():
    print("Finished all imports", flush=True)
    dict_of_hyperparameters, res_file_name, dataset_name, only_som, return_early, disable_model_training,\
        topic_model, only_tsne, force_reeval = get_arguments()
    print("Got parameters!", flush=True)

    if os.path.isfile(res_file_name) and not force_reeval:
        print("Found completely processed result. I will not reprocess and return.", flush=True)
        return

    start = time.time()
    filter_threshold_bow, needs_lemmatization, perform_standard_preprocessing, x, y = get_raw_dataset(dataset_name)
    print("Elapsed time for getting dataset: " + str(time.time() - start), flush=True)

    if dict_of_hyperparameters["lda"]["n_topics"] is None:
        dict_of_hyperparameters["lda"]["n_topics"] = len(np.unique(y))
    if dict_of_hyperparameters["lsi"]["n_topics"] is None:
        dict_of_hyperparameters["lsi"]["n_topics"] = len(np.unique(y))
    if dict_of_hyperparameters["nmf"]["n_topics"] is None:
        dict_of_hyperparameters["nmf"]["n_topics"] = len(np.unique(y))
    # For BERT a category is not the same as a topic
    if dict_of_hyperparameters["bert"]["n_categories"] is None:
        dict_of_hyperparameters["bert"]["n_categories"] = len(np.unique(y))

    dataset_dir = os.path.join("data", dataset_name)
    file_path = os.path.join(dataset_dir, dataset_name + "_words_list_" + str(len(x)) + ".pkl")
    print("Try to load dataset from: " + file_path, flush=True)
    if os.path.isfile(file_path):
        x = load_dataset_if_able(file_path)
    else:
        x = preprocess_texts(x, dataset_dir=dataset_dir, needs_lemmatization=needs_lemmatization,
                             needs_preprocessing=perform_standard_preprocessing, file_path=file_path)

    to_discard = [i for i, text in enumerate(x) if len(text) <= 1]
    x = [x[i] for i in range(len(x)) if i not in to_discard]
    y = np.array([y[i] for i in range(len(y)) if i not in to_discard])

    print("Successfully preprocessed dataset!", flush=True)
    print_current_memory_usage()

    if len(x) != len(y):
        raise ValueError("Please delete your data folder when working with slices of the original dataset")
    topic_layouts = create_all_topic_models(x, os.path.join("models", dataset_name),
                                            hyperparameters_lda=dict_of_hyperparameters['lda'],
                                            hyperparameters_lsi=dict_of_hyperparameters['lsi'],
                                            hyperparameters_nmf=dict_of_hyperparameters['nmf'],
                                            hyperparameters_bert=dict_of_hyperparameters['bert'],
                                            min_density=filter_threshold_bow,
                                            disable_model_training=disable_model_training,
                                            topic_model=topic_model)

    print("Successfully got all topic layouts", flush=True)
    print_current_memory_usage()

    if return_early:
        return

    res_df = None
    res_path = os.path.join("results", dataset_name)
    for experiment, high_embedding in topic_layouts.items():
        print("Processing experiment: " + str(experiment), flush=True)
        embeddings = create_layouts(high_embedding, y, res_path, dataset_name, experiment,
                                    umap_hyperparameters=dict_of_hyperparameters['umap'],
                                    tsne_hyperparameters=dict_of_hyperparameters['tsne'],
                                    som_hyperparameters=dict_of_hyperparameters['som'],
                                    mds_hyperparameters=dict_of_hyperparameters['mds'], only_som=only_som,
                                    only_tsne=only_tsne)

        start = time.time()
        res_df = evaluate_layouts(res_path, high_embedding, y, embeddings, dict_of_hyperparameters, old_df=res_df,
                                  res_file_name=res_file_name, topic_level_experiment_name=experiment,
                                  use_bow_for_comparison=True)
        print("Elapsed time for evaluation: " + str(time.time() - start), flush=True)

        # if "bow" not in experiment:
        #    res_df = evaluate_layouts(res_path, high_embedding, y, embeddings, dict_of_hyperparameters, old_df=res_df,
        #                              res_file_name=res_file_name, topic_level_experiment_name=experiment,
        #                              use_bow_for_comparison=False)
        print("Finished evaluation of experiment: " + str(experiment), flush=True)
        print_current_memory_usage()
        del embeddings
        gc.collect()
        print_current_memory_usage()

    os.replace(src=res_file_name.replace(".csv", "_partial.csv"), dst=res_file_name)


def get_raw_dataset(dataset_name, filter_threshold_bow=0.001, needs_lemmatization=True,
                    perform_standard_preprocessing=True):
    if dataset_name == "20_newsgroups":
        dataset = fetch_20newsgroups(subset="train", data_home=os.getcwd())
        x = dataset.data
        y = dataset.target
    elif dataset_name == "reuters":
        categories = reuters.categories()
        files = {category: reuters.fileids(category) for category in categories}
        unique_file_ids = []
        for file_ids in files.values():
            unique_file_ids.extend(file_ids)
        unique_file_ids = list(set(unique_file_ids))
        categories = []
        single_category_files = []
        for file_id in unique_file_ids:
            file_categories = reuters.categories(file_id)
            if len(file_categories) == 1:
                single_category_files.append(file_id)
                categories.append(file_categories[0])
        y = transform_categories_to_labels(dataset_name=dataset_name, y=categories)

        word_lists = [reuters.words(file_id) for file_id in single_category_files]
        x = [" ".join(word_list) for word_list in word_lists]
    elif dataset_name == "blogtext":
        blogtext_data_path = os.path.join("data", "blogtext", "blogtext.csv")
        if os.path.isfile(blogtext_data_path):
            df = pd.read_csv(blogtext_data_path)
        else:
            raise FileNotFoundError("First download blogtext.csv"
                                    " https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download"
                                    " and place it into the directory: " + blogtext_data_path +
                                    " We do this to avoid to have to interact with the kaggle API.")
        df = df.loc[df['topic'] != "indUnk"]
        df = df.loc[df['topic'] != "Student"]
        x = df["text"].to_list()
        y = df["topic"].to_numpy()
        del df
        gc.collect()

        print("Removed undefined and Student indices", flush=True)
        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
        desired_sample_size = 50000
        share = desired_sample_size / len(y)
        x, _, y, _ = train_test_split(x, y, train_size=share, random_state=0, shuffle=True, stratify=y)
        # unique_labels, counts = np.unique(y, return_counts=True)
        del _
        gc.collect()
    elif dataset_name == "github_projects":
        github_data_path = os.path.join("data", "github_projects")
        class_associations_file = os.path.join(github_data_path, "project_class_association.pkl")
        x = []
        y = []

        if not (os.path.isdir(github_data_path) or os.path.isfile(class_associations_file)):
            raise OSError("Can't find data directory at: " + github_data_path + " or class association file at: " +
                          class_associations_file)
        with open(class_associations_file, 'rb') as class_file:
            class_associations = pickle.load(class_file)
            for file in os.listdir(github_data_path):
                if class_associations_file.endswith(file) or "id_category_mapping" in file or dataset_name in file:
                    continue
                with open(os.path.join(github_data_path, file), 'rb') as pickle_file:
                    content = pickle.load(pickle_file)
                    x.append(content)
                project_name = file.replace(".pkl", "")
                y.append(class_associations[project_name])

        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
        needs_lemmatization = False
        perform_standard_preprocessing = False
        filter_threshold_bow = 0.005
    elif dataset_name == "emails":
        emails_root_dir = os.path.join("data", "emails")
        emails_data_path = os.path.join(emails_root_dir, "Data")
        if not os.path.isdir(emails_data_path):
            raise FileNotFoundError("First download data"
                                    " https://www.kaggle.com/datasets/dipankarsrirag/topic-modelling-on-emails?resource=download"
                                    " and place it into the directory: " + emails_root_dir +
                                    " We do this to avoid to have to interact with the kaggle API.")

        x = []
        y = []
        for category_dir in os.listdir(emails_data_path):
            class_email_data_path = os.path.join(emails_data_path, category_dir)
            for file in os.listdir(class_email_data_path):
                if not os.path.isfile(os.path.join(class_email_data_path, file)):
                    continue
                with open(os.path.join(class_email_data_path, file), encoding='windows-1252') as in_file:
                    x.append(in_file.read())
                    y.append(category_dir)

        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
    elif dataset_name == "ecommerce":
        ecommerce_path = os.path.join("data", "ecommerce", "ecommerceDataset.csv")
        if not os.path.isfile(ecommerce_path):
            raise FileNotFoundError("First download blogtext.csv"
                                    " https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification"
                                    " and place it into the directory: " + ecommerce_path +
                                    " We do this to avoid to have to interact with the kaggle API.")

        x = []
        y = []
        with open(ecommerce_path) as in_file:
            cf = csv.reader(in_file)
            for row in cf:
                x.append(row[1].strip())
                y.append(row[0].strip())

        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
    elif dataset_name == "seven_categories":
        seven_categories_root = os.path.join("data", "seven_categories")

        if not os.path.isdir(seven_categories_root):
            raise FileNotFoundError("First download data"
                                    " https://www.kaggle.com/datasets/deepak711/4-subject-data-text-classification"
                                    " and place it into the directory: " + seven_categories_root +
                                    " We do this to avoid to have to interact with the kaggle API.")

        restructure_target_root = os.path.join(seven_categories_root, "Physics_Biology_Geography_Accounts subject"
                                                                      " training data for text classification")
        restructure_target_dir = os.path.join(restructure_target_root, "train_data_final")
        if os.path.isdir(restructure_target_dir):
            for dir in os.listdir(restructure_target_dir):
                target_dir = os.path.join(seven_categories_root, dir)
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(os.path.join(restructure_target_dir, dir), target_dir)

            shutil.rmtree(restructure_target_root)

        x = []
        y = []
        for category_dir in os.listdir(seven_categories_root):
            class_path = os.path.join(seven_categories_root, category_dir)
            if os.path.isfile(class_path):
                continue
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path):
                    with open(os.path.join(class_path, file)) as in_file:
                        x.append(in_file.read())
                        y.append(category_dir)
                else:
                    class_path = file_path
                    for file in os.listdir(class_path):
                        file_path = os.path.join(class_path, file)
                        if os.path.isfile(file_path):
                            with open(os.path.join(class_path, file)) as in_file:
                                x.append(in_file.read())
                                y.append(category_dir)

        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
        filter_threshold_bow = 0.0005
    elif dataset_name == "github_authors":
        github_data_path = os.path.join("data", "github_authors")
        class_associations_file = os.path.join(github_data_path, "project_class_association.pkl")
        x = []
        y = []

        if not (os.path.isdir(github_data_path) or os.path.isfile(class_associations_file)):
            raise OSError("Can't find data directory at: " + github_data_path + " or class association file at: " +
                          class_associations_file)

        with open(class_associations_file, 'rb') as class_file:
            class_associations = pickle.load(class_file)
            for project_name in os.listdir(github_data_path):
                project_dir = os.path.join(github_data_path, project_name)
                if os.path.isfile(project_dir):
                    continue
                for file in os.listdir(project_dir):
                    with open(os.path.join(github_data_path, file), 'rb') as pickle_file:
                        content = pickle.load(pickle_file)
                        x.append(content)
                project_name = file.replace(".pkl", "")
                y.append(class_associations[project_name])

        y = transform_categories_to_labels(dataset_name=dataset_name, y=y)
        needs_lemmatization = False
        perform_standard_preprocessing = False
        filter_threshold_bow = 0.005
    else:
        NotImplementedError("Didn't recognize dataset '" + dataset_name + "' available options are: [20_newsgroups]")
    print("Got raw dataset!", flush=True)

    if isinstance(x[0], str):
        median = np.median(np.array([len(text.split()) for text in x]))
    else:
        median = np.median(np.array([len(text) for text in x]))
    print("Median length of text for dataset: " + dataset_name + " is " + str(median))
    return filter_threshold_bow, needs_lemmatization, perform_standard_preprocessing, x, y


def transform_categories_to_labels(dataset_name, y):
    unique_categories = np.unique(y)
    unique_categories = {category: i for i, category in enumerate(unique_categories)}
    y = np.array([unique_categories[category] for category in y])
    dataset_dir = os.path.join("data", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    mapping_path = os.path.join(dataset_dir, "id_category_mapping.pkl")
    if not os.path.isfile(mapping_path):
        with open(mapping_path, "wb+") as mapping_file:
            pickle.dump(unique_categories, mapping_file)
    return y


if __name__ == '__main__':
    main()
