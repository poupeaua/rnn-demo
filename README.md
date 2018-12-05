# RNN project on Natural Language Processing (NLP)

## Information

- Environnement : Python (2 or 3)
- Libraries : Tensorflow | Keras

## Train a model :

  Create a directory in data/ and then :
  Create a [name].txt file with at least 100K characters in it.
  Create an empty [name].h5 file to save the model configuration as it trains.

  Run nlp_train.py with the following arguments :
    --text [name].txt
    --save_model [name].h5

  N.B: The model is saved automatically after each epoch !

  It is obviously possible to train a pre-trained model by using the arguments --load_model.

## Test a model :

  Run nlp_test.py with the following arguments :
    --text [name].txt
    --load_model [name].h5

Enjoy !
