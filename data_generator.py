import itertools
import random
import string

import numpy as np

import encoding


def raw_word_generator():
    from nltk.corpus import webtext, reuters, brown, gutenberg

    return (
        w.lower()
        for w in itertools.chain(
            brown.words(), webtext.words(), reuters.words(), gutenberg.words(),
        )
        if w.isalnum()
    )


def add_typo(word, chance=0.2):
    if random.random() > chance:
        return word

    decision = random.random()

    if decision < 0.25:
        # Random char drop
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i+1:]

    if decision < 0.5:
        # Random char replace
        char = random.choice(string.ascii_lowercase)
        i = random.randint(0, len(word) - 1)
        return word[:i-1] + char + word[i:]

    if decision < 0.75:
        # Random char insert
        char = random.choice(string.ascii_lowercase)
        i = random.randint(0, len(word) - 1)
        return word[:i] + char + word[i:]

    # Random char switch
    i1 = random.randint(0, len(word) - 1)
    i2 = random.randint(max(0, i1 - 2), min(i1 + 2, len(word) - 1))
    c1 = word[i1]
    c2 = word[i2]
    word = word[:i1-1] + c2 + word[i1:]
    word = word[:i2-1] + c1 + word[i2:]
    return word


def get_most_common_words(count):
    from nltk import FreqDist

    frequency_list = FreqDist(i.lower() for i in raw_word_generator())
    frequency_list.most_common(count)

    return [w for w, _ in frequency_list.most_common(count)]


def infinitely_generate_random_common_words(count):
    words = get_most_common_words(count)

    while True:
        random.shuffle(words)
        for word in words:
            yield word


def store_w2v_model(word_count, filename):
    from gensim.models.keyedvectors import KeyedVectors

    model = KeyedVectors.load_word2vec_format(
        '../GoogleNews-vectors-negative300.bin',
        binary=True,
    )

    words = get_most_common_words(word_count)
    outputs = {}

    for word in words:
        if word not in model:
            print('Skipping', word)
            continue

        outputs[word] = model[word]

    np.savez_compressed(filename, outputs)


def load_w2v_model(filename):
    if not filename.endswith('.npz'):
        filename += '.npz'

    return np.load(filename)['arr_0'][()]


def word2vec_data_generator(
    w2v_model, char_ranks, max_word_length, batch_size,
):
    if type(w2v_model) is str:
        w2v_model = load_w2v_model(w2v_model)

    words = list(w2v_model.keys())

    n_chars = len(char_ranks)
    n_embedding_features = len(w2v_model[words[0]])

    batch_i = 0
    batch_x = None
    batch_y = None

    while True:
        for word in words:
            if len(word) > max_word_length:
                print('Word', word, 'too long')
                continue

            vector = w2v_model[word]

            # Add typos:
            max_n_typos = 2
            if len(word) <= 1:
                max_n_typos = 0
            elif len(word) <= 3:
                max_n_typos = 1
            elif len(word) <= 8:
                max_n_typos = 2
            else:
                max_n_typos = 3

            for i in range(max_n_typos):
                word = add_typo(word, 0.2 / (i + 1))

            if batch_x is None:
                batch_x = np.zeros(
                    (batch_size, max_word_length, n_chars),
                    dtype='float32'
                )
                batch_y = np.zeros(
                    (batch_size, n_embedding_features),
                    dtype='float32'
                )

            batch_x[batch_i][:] = encoding.get_character_features(
                [word], char_ranks, max_word_length
            )[0]
            batch_y[batch_i][:] = vector

            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0
