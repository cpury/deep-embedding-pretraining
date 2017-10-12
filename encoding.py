"""
encoding.py
Contains functions to encode data as one-hot vectors etc.
"""

import pickle
from collections import Counter

import numpy as np


def one_hot_to_index(v):
    """
    Given a one-hot vector, returns the index of the max value, or -1 if they
    are all zero.
    """
    if not np.any(v):
        return -1

    return np.argmax(v)


def features_to_text(vector, inverse_ranks, sep=''):
    """
    Given a single or a vector of one-hot vectors and the inverse ranks,
    restores the original text.
    """
    if not hasattr(vector, 'shape') or len(vector.shape) < 2:
        return inverse_ranks.get(one_hot_to_index(vector), '')

    return sep.join(
        inverse_ranks.get(one_hot_to_index(f), '')
        for f in vector
        if inverse_ranks.get(one_hot_to_index(f), '')
    )


def get_unique_words(rows, min_count=2):
    """Given a bunch of text rows, get all unique words that occur in it."""
    nlp = load_nlp()

    def word_iterator():
        for doc in nlp.pipe(rows):
            for word in doc:
                yield str(word).lower()

    return [w for w, c in Counter(word_iterator()).items() if c >= min_count]


def get_unique_characters(rows, min_count=2):
    """Given a bunch of text rows, get all unique chars that occur in it."""
    def char_iterator():
        for row in rows.values:
            for char in row:
                yield str(char).lower()

    return [
        c for c, count
        in Counter(char_iterator()).items()
        if count >= min_count
    ]


def store_ranks_to_file(ranks, inverse_ranks, filename):
    """
    Given a set of ranks and inverse ranks, stores them in a pickle file to
    be loaded later on.
    """
    content = {
        'ranks': ranks,
        'inverse_ranks': inverse_ranks,
    }

    with open(filename, 'wb') as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ranks_from_file(filename):
    """
    Given a pickled rank file, returns ranks and inverse_ranks.
    """
    with open(filename, 'rb') as f:
        content = pickle.load(f)

    return content['ranks'], content['inverse_ranks']


def get_char_ranks(char_list):
    """Given a list of characters, assigns each a unique integer and returns
    two mapping dictionaries."""
    char_ranks = {
        char: rank for rank, char in enumerate(char_list)
    }
    inverse_char_ranks = {
        rank: char for rank, char in enumerate(char_list)
    }
    # char_ranks[''] = 0
    # inverse_char_ranks[0] = ''

    return char_ranks, inverse_char_ranks


def get_character_features(rows, char_ranks, text_max_length, reverse=False):
    """Get one-hot character-level features."""
    n_rows = rows.shape[0] if hasattr(rows, 'shape') else len(rows)
    n_chars = len(char_ranks)

    output = np.zeros((n_rows, text_max_length, n_chars), dtype='float32')

    for i, row in enumerate(rows):
        if type(row) is not str:
            try:
                row = str(row)
            except Exception as e:
                print('Error converting', e)
                continue

        if reverse:
            row = row[::-1]

        shift = 0
        for j, char in enumerate(row[:text_max_length]):
            char_rank = char_ranks.get(char)
            if char_rank is None:
                # Skip this one:
                shift += 1
                continue

            output[i, j - shift, char_rank] = 1.

    return output


def get_character_mappings(rows, char_min_count=0):
    unique_chars = get_unique_characters(rows, min_count=char_min_count)
    return get_char_ranks(unique_chars)
