#!/usr/bin/env python
# coding: utf-8
import warnings
import pandas as pd
import shutil
import git

from code_preprocessing import *
from commons import write_list, split_file
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

ignoreDirs = ['.git', 'build', 'tests', 'docs']


def main():
    os.makedirs(os.path.join("data", "github_projects"), exist_ok=True)

    df_projects = pd.read_csv(os.path.join(os.getcwd(), "Github_projects.csv"), index_col=0)
    project_list = df_projects["project"].tolist()
    list_folder_names = []
    list_folder_names_pick = []

    for url in tqdm(project_list):
        folder_name = url.split("/")[-1].replace(".git", "")
        file_name_pick = os.path.join("data", "github_projects", folder_name + ".pkl")
        if os.path.isfile(file_name_pick):
            continue

        try:
            try:
                print("Cloning " + url)
                git.Git('/temp').clone(url, depth=1)  # the folder temp does not work
                list_folder_names.append(folder_name)

                list_folder_names_pick.append(file_name_pick)
                print(folder_name)

                print("Collecting files")
                files = []

                for dirpath, dirnames, filenames in os.walk(folder_name):
                    if any([d in dirpath for d in ignoreDirs]):
                        continue
                    files.extend(os.path.join(dirpath, name) for name in filenames if any(name.endswith(ext)
                                                                                          for ext in filterExtensions))
                print("Number of files: " + str(len(files)))
                if len(files) == 0:
                    warnings.warn("Couldn't get any files for: " + url)
                    continue

                print("Parsing files")
                corpus = []

                for file in files:
                    with open(file, 'r', encoding='utf8', errors='ignore') as f:
                        corpus.append(f.read())

                corpus_new = []
                for doc in tqdm(corpus):
                    if len(doc) < 20000:
                        corpus_new.append(doc)
                    else:
                        r = split_file(doc, chunksize=5000)
                        for doc_split in r:
                            corpus_new.append(doc_split)
                corpus = corpus_new

                corpus = corpus_split(corpus)

                # Step 1:
                print("Step 1: Removal of symbols")
                corpus = remove_signs(corpus)

                # Step 2:
                print("Step 2: Splitting at signs")
                print("Splitting at doublepoint")
                corpus = corpus_doublepoint_split(corpus)
                print("Splitting at dot")
                corpus = corpus_dot_split(corpus)
                print("Splitting at slash")
                corpus = corpus_slash_split(corpus)
                print("Splitting at minus")
                corpus = corpus_minus_split(corpus)
                print("Splitting at underscore")
                corpus = corpus_underscore_split(corpus)
                print("cleaning")
                corpus = corpus_withoutnumbers(corpus)

                # Step 3:
                print("Step 3: Camel case split")
                corpus = corpus_camel_case_split(corpus)

                # Step 4:
                print("Step 4: Removal of keywords of the programming language")
                corpus = remove_programming_words(corpus)

                # Step 5:
                print("Step 5: Removal of stopwords of the natural language")
                corpus = remove_stopwords(corpus)

                # Step 6:
                print("Step 6: Lemmatization")
                corpus = lemmatization(corpus)

                print(corpus[0])

                corpus_single_file = merge_corpus(corpus)
                write_list(corpus_single_file, file_name_pick)
            except:
                print("Removing folder")
                if os.path.exists(folder_name):
                    shutil.rmtree(folder_name, onerror=clear_readonly_flag)
                continue
        finally:
            print("Removing folder")
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name, onerror=clear_readonly_flag)


if __name__ == "__main__":
    main()
