# coding : utf

import argparse
import numpy as np


def analyse_nlp_train():
    """
        Function used to analyse the arguments provided by the user to execute
        the nlp_train.py file.

        Returns:
            config (dict) : dictionnary where a value is assigned to all the
                required variables.
    """
    parser = argparse.ArgumentParser()
    # the first two arguments are REQUIRED
    parser.add_argument('--text', type=str, help=
        "String defining the .txt database file.", required=True)
    parser.add_argument('--save_model', type=str, help=
        "String defining the .h5 file where the configuration of the trained \
        model will be stored.", required=True)
    parser.add_argument('--loaded_model', type=str, help=
        "String defining the .h5 file that contains the configuration of a \
        pre-trained model to re-train it and make it even better.")
    parser.add_argument('--gen_text', type=str, help=
        "String defining the .txt file where the text generated after each \
        epoch will be stored.")
    parser.add_argument('--cell', type=str, default="LSTM", choices=
        ["SimpleRNN", "LSTM", "GRU"], help=
        "String defining the type of cell use in the RNN.")
    parser.add_argument('--hidden_state_size', type=int, default=128, help=
        "String defining the size of the hidden state layer in each RNN cell. \
        The bigger the hidden state, the better the memory of the RNN.")
    parser.add_argument('--time_steps', type=int, default=40, help=
        "Integer defining the size of the RNN for the training (number \
        of consecutive RNN cell in the model)")
    parser.add_argument('--step', type=int, default=3, help=
        "Integer defining the step each sentence and expected character for \
        the training.")
    parser.add_argument('--epochs', type=int, default=20, help=
        "Integer defining the number of iteration on the whole provided text to\
         train the model.")
    parser.add_argument('--batch_size', type=int, default=128, help=
        "Integer defining the size of a batch.")
    parser.add_argument('--nb_char', type=int, default=400, help=
        "Integer defining the number of characters generated after each epochs \
        to follow the model optimization.")
    parser.add_argument('--lr', type=float, default=0.01, help=
        "Float number that defines the learning rate.")
    parser.add_argument('--temperature', type=float, default=0.2, help=
        "Float number that defines the diversity of the generated characters. \
        The bigger the temperature, the higher rate of diversity.")
    config = vars(parser.parse_args())
    return config


def analyse_nlp_test():
    """
        Function used to analyse the arguments provided by the user to execute
        the nlp_test.py file.

        Returns:
            config (dict) : dictionnary where a value is assigned to all the
                required variables.
    """
    parser = argparse.ArgumentParser()
    # the first two arguments are REQUIRED
    parser.add_argument('--text', type=str, help=
        "String defining the .txt file as the text database.", required=True)
    parser.add_argument('--loaded_model', type=str, help=
        "String defining the model configuration .h5 file" , required=True)
    parser.add_argument('--nb_char', type=int, default=400, help=
        "Integer defining the number of characters generated during the \
        test session on the screen.")
    parser.add_argument('--temperature', type=float, default=0.2, help=
        "Float number that defines the diversity of the generated characters. \
        The bigger the temperature, the higher rate of diversity.")
    parser.add_argument('--time_steps', type=int, default=40, help=
        "")
    config = vars(parser.parse_args())
    return config


def sample(preds, temperature=1.0):
    """
        Function used to generate a one-hot vector given a "probability" vector.

        Arguments:
            preds (array) : float array where the number are in [0, 1].
            temperature (float) : float that defines the diversity of the
                generated characters.

        Returns:
            np.argmax(probas) (array) : one-hot vector where the one is the
                position of the elected character.
    """
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
