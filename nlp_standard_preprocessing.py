import os
import pickle
import re

import gensim
import spacy
from nltk.corpus import stopwords, words
import multiprocessing

use_python_parallelism = False
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
word_list = words.words()
data_base_path = "data"


def has_vowel(word):
    sum_vowels = 0
    vowels = 'aeiouäöüy'
    for vowel in vowels:
        sum_vowels += word.count(vowel)
    if sum_vowels > 0:
        return True
    else:
        return False


def is_english(word):
    return word in word_list


def remove_symbols(cur_body):
    cur_body = re.sub(r'\d', '', cur_body)
    return cur_body.replace("\"", "").replace("-", " ").replace("_", " ").replace(",", "").replace("\'", ""). \
        replace("\t", " ").replace("/", "").replace("\\", "").replace(">", "").replace("<", "").replace("(", "").\
        replace(")", "").replace("]", " ").replace("[", "").replace("{", "").replace("}", "")


def preprocess_single_text(text, needs_lemmatization=True):
    cur_body = text.lower()
    cur_body = remove_symbols(cur_body)
    all_words = gensim.utils.simple_preprocess(cur_body, deacc=True)
    all_words_preprocessed = []

    if len(all_words) > 1000 and needs_lemmatization:
        for i in range(int(len(all_words) / 1000)):
            words = all_words[i*1000:min((i+1)*1000, len(all_words))]
            words = preprocess_word_list(words, needs_lemmatization)
            all_words_preprocessed.extend(words)
    else:
        all_words_preprocessed.extend(preprocess_word_list(words=all_words, needs_lemmatization=needs_lemmatization))
    return all_words_preprocessed


def preprocess_word_list(words, needs_lemmatization):
    if needs_lemmatization:
        text = " ".join(words)
        tokens = nlp(text)
        words = [token.lemma_ for token in tokens]
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if has_vowel(word)]
    words = [word for word in words if is_english(word)]
    return words


def load_dataset_if_able(file_path):
    if file_path is not None:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as fp:
                words_list = pickle.load(fp)
                print("Found already preprocessed dataset! I will load and return early", flush=True)
                print("Length of word list: " + str(len(words_list)), flush=True)
                return words_list
        else:
            return None


def preprocess_texts(texts, dataset_dir=None, file_path=None, needs_lemmatization=True, needs_preprocessing=True):
    os.makedirs(dataset_dir, exist_ok=True)

    if needs_preprocessing:
        if use_python_parallelism:
            with multiprocessing.Pool(processes=10) as pool:
                parameters = [(text, needs_lemmatization) for text in texts]
                words_list = pool.starmap(preprocess_single_text, parameters)
        else:
            words_list = [preprocess_single_text(text, needs_lemmatization=needs_lemmatization) for text in texts]
    else:
        words_list = texts

    if file_path is not None and not os.path.isfile(file_path):
        with open(file_path, 'wb+') as fp:
            pickle.dump(words_list, fp)

    return words_list
