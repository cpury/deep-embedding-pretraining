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
    x_test, y_test,
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
            validation_data=(x_test, y_test),
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


def test_model(model, ranks, inverse_ranks, w2v_model, count, scaler=None):
    max_word_length = model.input_shape[1]
    x_test, y_test = data_generator.build_word2vec_data(
        w2v_model, ranks, max_word_length, scaler=scaler, count=count
    )

    n_test_data = len(x_test)
    count = min(count, n_test_data)
    indexes = random.sample(range(n_test_data), count)

    test_rows = x_test[indexes]
    predictions = model.predict(test_rows)
    targets = y_test[indexes]

    for i in range(count):
        print('{}: {} -> {:.8f}'.format(
            i,
            encoding.features_to_text(test_rows[i], inverse_ranks),
            np.linalg.norm(targets[i] - predictions[i]),
        ))


def main():
    max_word_length = 32
    batch_size = 64
    depth = 2
    hidden_size = 300

    training_steps_per_epoch = round(9000 / 64)
    epochs = 100

    w2v_model = data_generator.load_w2v_model('10k.npz')
    train_w2v_model, test_w2v_model = data_generator.split_w2v_model(w2v_model)

    ranks, inverse_ranks = encoding.load_ranks_from_file('ranks.pickle')

    scaler = data_generator.fit_scaler_on_w2v_data(train_w2v_model)

    generator = data_generator.word2vec_data_generator(
        train_w2v_model, ranks, max_word_length, batch_size, scaler=scaler
    )
    x_test, y_test = data_generator.build_word2vec_data(
        test_w2v_model, ranks, max_word_length, scaler=scaler
    )

    model = build_model(max_word_length, len(ranks), depth, hidden_size, 300)

    print()
    model.summary()
    print()

    print()
    test_model(model, ranks, inverse_ranks, test_w2v_model, 10, scaler=scaler)
    print()

    train_from_generator(
        model,
        generator, training_steps_per_epoch,
        x_test, y_test,
        epochs, batch_size,
    )

    print()
    test_model(model, ranks, inverse_ranks, test_w2v_model, 10, scaler=scaler)
    print()


if __name__ == '__main__':
    main()
