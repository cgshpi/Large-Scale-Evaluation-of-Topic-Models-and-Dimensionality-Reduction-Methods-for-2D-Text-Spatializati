import os
import re
import stat

import nltk
import spacy
from gensim.utils import simple_preprocess
from tqdm import tqdm
from ignore_words import get_ignore_words

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Do lemmatization keeping only noun, adj, vb, adv
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])



def clean_aftersplit(corpus):
    corpus_new = []
    for document in corpus:
        document_new = []
        for word in document:
            if word.isdigit() == False:
                if len(word) > 1:
                    document_new.append(word)
        corpus_new.append(document_new)
    return corpus_new


def remove_signs(corpus):
    # remove symbols that carry no meaning
    signs = ['{', '}', '!', '\"', '§', '$', '%', '&', '(', ')', '=', '?', '\\',
             '*', '+', '\'', '#', ';', ',', '<', '>', '|', '^', '°', '[', ']']

    corpus_withoutSigns = []
    for s in corpus:
        s_new = []
        for word in s:
            word_new = ''
            for letter in word:
                if letter not in signs:
                    word_new = word_new + letter
            if word_new != '':
                s_new.append(word_new)
        corpus_withoutSigns.append(s_new)
    return corpus_withoutSigns


# splitting identifier names at symbols
# Splitting at ":"
def doublepoint_split(str):
    return str.split(":")


def corpus_doublepoint_split(corpus):
    corpus_doublepoint_split = []
    for document in corpus:
        document_doublepoint_split = []
        for term in document:
            document_doublepoint_split = document_doublepoint_split + doublepoint_split(term)
        corpus_doublepoint_split.append(document_doublepoint_split)
    return corpus_doublepoint_split


# Splitting at "."
def dot_split(str):
    return str.split(".")


def corpus_dot_split(corpus):
    corpus_dot_split = []
    for document in corpus:
        document_dot_split = []
        for term in document:
            document_dot_split = document_dot_split + dot_split(term)
        corpus_dot_split.append(document_dot_split)
    return corpus_dot_split


# Splitting at "/"
def slash_split(str):
    return str.split("/")


def corpus_slash_split(corpus):
    corpus_slash_split = []
    for document in corpus:
        document_slash_split = []
        for term in document:
            document_slash_split = document_slash_split + slash_split(term)
        corpus_slash_split.append(document_slash_split)

    return corpus_slash_split


# Splitting at "-"
def minus_split(str):
    return str.split("-")


def corpus_minus_split(corpus):
    corpus_minus_split = []
    for document in corpus:
        document_minus_split = []
        for term in document:
            document_minus_split = document_minus_split + minus_split(term)
        corpus_minus_split.append(document_minus_split)
    return corpus_minus_split


def underscore_split(str):
    return str.split("_")


def corpus_underscore_split(corpus):
    corpus_underscore_split = []
    for document in corpus:
        document_underscore_split = []
        for term in document:
            document_underscore_split = document_underscore_split + underscore_split(term)
        corpus_underscore_split.append(document_underscore_split)
    return corpus_underscore_split


def corpus_withoutnumbers(corpus):
    corpus_new = []
    for document in corpus:
        document_new = []
        for word in document:
            if word.isdigit() == False:
                if len(word) > 1:
                    document_new.append(word)
        corpus_new.append(document_new)
    return corpus_new


# camel case split
def camel_case_split(str):
    str = str[0].upper() + str[1:]
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)
    l_output = []
    for word in l:
        l_output.append(word.lower())
    return l_output


# def corpus_camel_case_split(corpus):
#    corpus_camel_case_split = []
#    for document in corpus:
#        document_camel_case_split = []
#        for term in document:
#            document_camel_case_split = document_camel_case_split + camel_case_split(term)
#        corpus_camel_case_split.append(document_camel_case_split)#
#
#    corpus_new = []
#    for document in corpus_camel_case_split:
#        document_new = []
#        i = 0
#        for word in document:
#           if len(word) > 1:
#               document_new.append(word)
#               i = i + 1
#           if len(word) == 1:
#               document[i + 1] = word + document[i + 1]
#               i = i + 1
#       corpus_new.append(document_new)
#   return corpus_new


# remove keywords of programming language
def remove_programming_words(corpus):
    ignoreWords = get_ignore_words()
    corpus_withoutProgrammingWords = []
    for s in corpus:
        s_new = []
        for word in s:
            if word not in ignoreWords:
                s_new.append(word)
        corpus_withoutProgrammingWords.append(s_new)
    return corpus_withoutProgrammingWords


# remove stopwords
def corpus_remove_stopwords(corpus):
    return [[word for word in doc if word not in stop_words] for doc in corpus]


def full_preprocessing(corpus):
    c = remove_signs(corpus)
    c = corpus_doublepoint_split(c)
    c = corpus_underscore_split(c)
    c = corpus_minus_split(c)
    c = corpus_slash_split(c)
    c = corpus_dot_split(c)
    c = corpus_withoutnumbers(c)
    c = corpus_camel_case_split(c)
    c = remove_programming_words(c)
    c = corpus_remove_stopwords(c)
    return c


filterExtensions = {
    '.c', '.cats', '.h', '.idc',  # C

    '.cpp', '.c++', '.cc', '.cxx', '.h', '.h++', '.hh', '.hpp', '.hxx', '.inc',
    '.inl', '.ino', '.ipp', '.re', '.tcc', '.tpp',  # C++

    '.java',  # Java

    '.js', '._js', '.bones', '.es', '.es6', '.frag', '.gs', '.jake', '.jsb',
    '.jscad', '.jsfl', '.jsm', '.jss', '.mjs', '.njs', '.pac', '.sjs', '.ssjs',
    '.xsjs', '.xsjslib',  # JavaScript

    '.php', '.aw', '.ctp', '.fcgi', '.inc', '.php3', '.php4', '.php5', '.phps',
    '.phpt',  # PHP

    '.py', '.blz', '.cgi', '.fcgi', '.gyp', '.gypi', '.lmi', '.py3', '.pyde',
    '.pyi', '.pyp', '.pyt', '.pyw', '.rpy', '.spec', '.tac', '.wsgi', '.xpy',
    # Python

    '.rb', '.builder', '.eye', '.fcgi', '.gemspec', '.god', '.jbuilder',
    '.mspec', '.pluginspec', '.podspec', '.rabl', '.rake', '.rbuild', '.rbw',
    '.rbx', '.ru', '.ruby', '.spec', '.thor', '.watchr',  # Ruby

    ".go",  # Go programming language

    ".cs",  # C Sharp

    ".move",  # Move programming language

    ".rs",  # Rust programming language

    ".sol"  # Solidity programming language
}


def clear_readonly_flag(func, path, exInfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def corpus_split(corpus):
    corpus_split = []
    for document in corpus:
        corpus_split.append(document.split())
    return corpus_split


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        text = " ".join(sent)
        if len(text) > 1000000:
            text = text[:1000000]
        doc = nlp(text)
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def merge_corpus(corpus):
    result = []
    for doc in corpus:
        result += doc
    return result


def corpus_camel_case_split(corpus):
    corpus_camel_case_split = []
    for document in tqdm(corpus):
        document_camel_case_split = []
        for term in document:
            document_camel_case_split = document_camel_case_split + camel_case_split(term)
        corpus_camel_case_split.append(document_camel_case_split)

    return corpus_camel_case_split
