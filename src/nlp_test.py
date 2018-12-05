# coding : utf

from keras.models import load_model

import sys
import numpy as np
import random

from utils import *


def nlp_test():
    """
        Function used to test a pre-trained model
    """
    config = analyse_nlp_test()
    print("\nThe configuration for the test session is :\n", config)

    # load the text on which the model had been trained
    with open(config["text"]) as f:
        text = f.read().lower()

    # extract some variables needed for the test
    time_steps = config["time_steps"]
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print('total chars:', vocab_size)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # load the pre-trained model
    model = load_model(config["load_model"])

    # beggining of the test
    start_index = random.randint(0, len(text) - time_steps - 1)

    generated = '\n'
    sentence = text[start_index: start_index + time_steps]
    generated += sentence
    print('\n----- Generating with seed: \n\n' + sentence)
    print('\n ----- Text Generation ----- ')
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
    print()


if __name__ == '__main__':
    nlp_test()
