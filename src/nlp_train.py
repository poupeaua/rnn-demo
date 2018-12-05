# coding : utf

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard, \
                            ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys

from utils import *

def nlp_train():
    """
        Function used to trained a model
    """
    config = analyse_nlp_train()
    print('\nThe configuration for the training session is :\n', config)

    print("\n1) Getting the text sample...")
    with open(config["text"]) as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print('total chars:', vocab_size)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of time_steps characters
    time_steps = config["time_steps"]
    step = config["step"]
    sentences = []
    next_chars = []
    for i in range(0, len(text) - time_steps, step):
        sentences.append(text[i: i + time_steps])
        next_chars.append(text[i + time_steps])
    training_size = len(sentences)
    print('nb sequences:', training_size)

    print('\n2) Vectorization...\n')
    # creating empty placeholders for x and y, the expected character
    x = np.zeros((training_size, time_steps, vocab_size), dtype=np.bool)
    y = np.zeros((training_size, vocab_size), dtype=np.bool)

    # associate each character to a one-hot vector of size vocab_size
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # print some stuff to better understand the proccess of Vectorization
    # and spitting between x and y
    for _ in range(3):
        index = random.randint(0, len(sentences)-1)
        for t in range(time_steps):
            print(indices_char[np.argmax(x[index,t,])], end='')
        print("\nThe expected character is -> ",
            indices_char[np.argmax(y[index,])], "\n")

    # size of the hidden_state
    hidden_state_size = config["hidden_state_size"]

    print('\n3) Build model...\n')

    if config["load_model"] is None:
        # build a fresh new model
        model = Sequential()
        if config["cell"] == "SimpleRNN":
            model.add(SimpleRNN(hidden_state_size,
                input_shape=(time_steps, vocab_size)))
        elif config["cell"] == "LSTM":
            model.add(LSTM(hidden_state_size,
                input_shape=(time_steps, vocab_size)))
        elif config["cell"] == "GRU":
            model.add(GRU(hidden_state_size,
                input_shape=(time_steps, vocab_size)))
        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))
    else:
        # re-use a pre-trained model to train it more
        print("Loaded model from the", config["load_model"], "file.")
        model = load_model(config["load_model"])

    # summarize all the trainable parameters of the model and its configuration
    model.summary()

    # explains how we are going to make the model better
    optimizer = RMSprop(lr=config["lr"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # do some stuff at the end of each epoch
    def on_epoch_end(epoch, logs):
        """
            Function invoked at the end of each epoch.
        """
        print()
        generated = '\n\n----- Generating text after Epoch: ' +  str(epoch)

        start_index = random.randint(0, len(text) - time_steps - 1)

        generated += '\n\n'
        sentence = text[start_index: start_index + time_steps]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for _ in range(config["nb_char"]):
            x_pred = np.zeros((1, time_steps, vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, config["temperature"])
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        if config["gen_text"] is not None:
            # open in a = append mode because we want to follow the progression
            file = open(config["gen_text"], "a")
            file.write(generated)
            print("\n\nINFO : The previous generated text has been successfully saved in the",
                config["gen_text"], "file.")
        print()

    callbacks_list = [ModelCheckpoint(config["save_model"]),
                    TensorBoard(log_dir='./logs'),
                    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7),
                    LambdaCallback(on_epoch_end=on_epoch_end)]

    print("\n4) Training the model...\n")
    model.fit(x, y,
              batch_size=config["batch_size"],
              epochs=config["epochs"],
              callbacks=callbacks_list)

if __name__ == '__main__':
    nlp_train()
