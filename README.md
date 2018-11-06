# LT 2316 H18 Course Project
Course project for [Asad Sayeed's](https://asayeed.github.io) machine learning class 
in the University of Gothenburg's Masters of Language Technology programme.

### Dataset and GloVe model
The dataset employed to train the models has to be 
[downloaded](https://www.kaggle.com/stephanerappeneau/350-000-movies-from-themoviedborg#AllMoviesDetailsCleaned.csv) 
and placed inside a folder named "dataset/" in the root folder of the project.\
The dataset is the csv file called AllMoviesDetailsCleaned.csv.

The GloVe model for the word embeddings layer of the CNN model has to be 
[downloaded](https://nlp.stanford.edu/projects/glove/)
and placed inside a folder named "glovedata/" in the root folder of this project.
The model file employed is inside the file glove.6B.zip and it is called glove.6B.100d.txt.

### Instructions to run the code.
For running train.py, the following arguments are needed:\
python3 train.py --model|-M trainingsize modelfile
- model: the type of model to train. Two models are available:
    - -Mrnn to train a Recurrent Neural Network model.
    - -Mcnn to train a Convolutional Neural Network model.
- trainingsize: Name of the file to use for training the model, corresponding to 
the size of the data. It has to be one of the following:
    - train30.pickle: 30% of the dataset.
    - train50.pickle: 50% of the dataset.
    - train70.pickle: 70% of the dataset.
- modelfile: path and name of the output model file to save inside the models/ folder. 
For instance:
    - rnn/lstm-train70.h5 will save the model "lstm-train70.h5" inside the folder 
    models/rnn/.

For running test.py, the following arguments are needed:\
python3 test.py --mode|-M modelfile --synopsis|-S
- mode: the mode to run the test. Two modes are available:
    - -Mevaluate|-ME: evaluate the model, print the metrics on screen and save 
    history plots inside the plots/ folder.
    - -Mpredict|-MP: use the model to predict genres of a given synopsis. Prints the 
    predictions and scores on screen.
- modelfile: path and name of the output model file to load from inside the models/ folder. 
For instance:
    - rnn/lstm-train70.h5 will load a model "lstm-train70.h5" from inside the folder 
    models/rnn/.
- synopsis: a synopsis to predict its genres. Needed only if mode is set to predict.