import pickle

import numpy as np


def transform_sparse_model_to_dense_matrix(bow_corpus, vector_length):
    dense_matrix = []
    for tokenized_line in bow_corpus:
        cur_pos = 0
        dense_vector = []
        for i in range(0, vector_length):
            if len(tokenized_line) > 0 and tokenized_line[cur_pos][0] == i:
                dense_vector.append(tokenized_line[cur_pos][1])
                cur_pos = min(cur_pos + 1, len(tokenized_line) - 1)
            else:
                dense_vector.append(0)
        dense_matrix.append(np.array(dense_vector))
    dense_matrix = np.array(dense_matrix)
    return dense_matrix


def write_list(a_list, file_path):
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb+') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def split_file(file, chunksize):
    chunks = [file[x:x + chunksize] for x in range(0, len(file), chunksize)]
    return chunks
