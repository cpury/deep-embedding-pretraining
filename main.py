# To make get similar random data splits etc:
# https://keras.io/getting-started/faq/#how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch
RANDOM_SEED = 1

import numpy as np
np.random.seed(RANDOM_SEED)
import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
import random as rn
rn.seed(RANDOM_SEED)

import random

from keras import backend as K

import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.layers.core import Masking
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization

import encoding
import data_generator


def build_model(
    input_length, char_count, depth, hidden_size, output_size, dropout=0.5
):
    input_shape = (input_length, char_count)

    model = Sequential()

    model.add(Masking(input_shape=input_shape))

    for i in range(depth):
        layer_kwargs = {}
        layer_kwargs['return_sequences'] = i < (depth - 1)
        if input_shape and i == 0:
            layer_kwargs['input_shape'] = input_shape

        model.add(LSTM(
            hidden_size,
            **layer_kwargs
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(output_size))
    model.add(Activation('tanh'))

    model.compile(
        optimizer=Adam(clipnorm=5.),
        loss='mse',
    )

    return model


def train_from_generator(
    model,
    training_generator, training_steps_per_epoch,
    epochs, batch_size,
    **kwargs
):
    history = None

    try:
        history = model.fit_generator(
            training_generator,
            steps_per_epoch=training_steps_per_epoch,
            max_queue_size=(2 * batch_size),
            epochs=epochs,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    patience=100,
                    monitor='loss',
                ),
                ModelCheckpoint(
                    'model.h5',
                    monitor='loss',
                    save_best_only=True,
                ),
                CSVLogger('log.csv'),
            ],
            **kwargs
        )
    except KeyboardInterrupt:
        print(' Got Sigint')

    return history


def test_model(model, char_ranks, w2v_model, scaler=None):
    max_word_length = model.input_shape[1]
    test_words = ['apple', 'dog', 'germany', 'france', 'hey']

    for word in test_words:
        target = w2v_model[word]

        if scaler:
            target = scaler.transform(target.reshape(1, -1))[0]

        prediction = model.predict(
            encoding.get_character_features(
                [word], char_ranks, max_word_length
            )
        )[0]
        print(word, np.linalg.norm(target - prediction))


def main():
    max_word_length = 32
    batch_size = 64
    depth = 2
    hidden_size = 300

    training_steps_per_epoch = round(9000 / 64)
    epochs = 100

    w2v_model = data_generator.load_w2v_model('10k.npz')
    ranks, inverse_ranks = encoding.load_ranks_from_file('ranks.pickle')

    scaler = data_generator.fit_scaler_on_w2v_data(w2v_model)

    generator = data_generator.word2vec_data_generator(
        w2v_model, ranks, max_word_length, batch_size, scaler=scaler
    )

    model = build_model(max_word_length, len(ranks), depth, hidden_size, 300)

    print()
    model.summary()
    print()

    print()
    test_model(model, ranks, w2v_model, scaler=scaler)
    print()

    train_from_generator(
        model,
        generator, training_steps_per_epoch,
        epochs, batch_size,
    )

    print()
    test_model(model, ranks, w2v_model, scaler=scaler)
    print()


if __name__ == '__main__':
    main()
