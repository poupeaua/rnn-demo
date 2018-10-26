RNN projet sur le Natural Language Processing (NLP)

Créer un répertoire dans data puis :

1) Entrainer un modèle :
  Créer un fichier texte .txt contenant au moins 100K caracteres.
  Créer un fichier .h5 vide pour enregistrer la configuration du modèle
  après entrainement.

  Lancer le programme python3 nlp_train.py avec pour arguments:
    --text "nom du fichier .txt"
    --save_model "nom du fichier .h5"

  N.B: Le modèle est sauvegardé à chaque epoch !

  Il est bien entendu possible d'entrainer un modèle pré-entrainer, voir
  l'argument --loaded_model.



2) Tester son modèle :
  Lancer le programme python3 nlp_test.py avec pour arguments:
    --text "nom du fichier .txt"
    --loaded_model "nom du fichier .h5"

  Enjoy !
