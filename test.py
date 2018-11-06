# test.py
# This is the script to test and evaluate a models for text classification for
# the AllMoviesDetailsCleaned.csv dataset.

# Since it is running in the server, the matplotlib.use('Agg') is needed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from train import load_file, prepare_input_data, prepare_output_categories
from keras.models import load_model
from nltk.tokenize import word_tokenize


def evaluation_print(model):
    """
    Evaluates a model with a test dataset and prints the loss and accuracy.
    :param model: the model to evaluate.
    """
    # Load the test dataset
    if os.path.isfile('splits/test20.pickle') and \
            os.path.isfile('splits/all_genres.pickle'):
        print("Loading test dataset (test20.pickle) and the list of genres"
              "(all_genres.pickle).")
        test = load_file('splits/test20.pickle')
        all_genres = load_file('splits/all_genres.pickle')

    else:
        print("There is no test dataset and genres list files in the splits/ "
              "folder.\nClosing the application.")
        exit(0)

    # Prepare the test data
    print("Preparing the test data to evaluate...")
    syns_pad = prepare_input_data(test)[0]
    gens_vec = prepare_output_categories(test, all_genres)
    print("Done!\n")

    # Evaluate and print metrics values
    print("Metrics values:")
    evaluation = model.evaluate([syns_pad], [gens_vec])
    metrics = ["Loss", "Accuracy"]
    for name, value in zip(metrics, evaluation):
        print("%s = %f" % (name, value))


def plot_history_save(modelfile):
    """
    Design the accuracy and loss plots and save them into plots/ folder.
    :param modelfile: the path where the model file was saved.
    """
    history_path = "models/" + modelfile.replace(".h5", "-hist.pickle")
    model_history = load_file(history_path)
    acc_plot_path = "plots/" + modelfile.replace(".h5", "-acc.png")
    loss_plot_path = "plots/" + modelfile.replace(".h5", "-loss.png")

    print("\nBuilding and saving plots...")
    # Plot history for accuracy
    plt.plot(model_history['acc'])
    plt.plot(model_history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(acc_plot_path, dpi=150)

    plt.clf()

    # Plot history for loss
    plt.plot(model_history['loss'])
    plt.plot(model_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(loss_plot_path, dpi=150)
    print("Plots saved in the plots/ folder.")


def predict_genres_print(model, synopsis):
    """
    Loads the genres list, predicts the genres of a given synopsis and prints
    the prediction.
    :param model: the model to do the prediction.
    :param synopsis: the str input for the prediction.
    """

    # Load all_genres list
    if os.path.isfile('splits/all_genres.pickle'):
        print("Loading the list of all genres (all_genres.pickle).")
        all_genres = load_file('splits/all_genres.pickle')
    else:
        print("There is no genres list file in the splits/ folder.\n"
              "Closing the application.")
        exit(0)

    print("\nPrediction:")
    # Tokenize the synopsis and filter punctuation
    synopsis_tok = word_tokenize(synopsis.lower())
    synopsis_list = list(filter(
        lambda word: word not in '!"#$%&(--)*+,``\'\'-./:;<=>?@[\]^_{|}~...',
        synopsis_tok))
    # Sequence and pad the synopsis
    test_pad = prepare_input_data([(synopsis_list, [])])[0]

    # Predict
    prediction = model.predict(test_pad)

    # Decode categories
    gens_pred = \
        [x for x in sorted(zip(prediction[0], all_genres), reverse=True)]

    # Print results
    print("Synopsis:", synopsis)
    print("Genres sorted by their scores:")
    for i, j in gens_pred:
        print("\t%s: %f" % (j, i))


if __name__ == "__main__":
    parser = ArgumentParser("python3 test.py")
    parser.add_argument('-M', '--mode', type=str,
                        help="The mode of test (REQUIRED):\n"
                             "\tevaluate | E = evaluate the model and save "
                             "history plots.\n"
                             "\tpredict | P = use the model to predict genres "
                             "of a given synopsis.",
                        required=True)
    parser.add_argument('modelfile', type=str,
                        help="Path inside the model/ folder of the model file "
                             "to evaluate or predict.")
    parser.add_argument('-S', '--synopsis', type=str,
                        help="a synopsis to predict its genres", required=False)
    args = parser.parse_args()

    # Load the model
    model = load_model("models/" + args.modelfile)

    if args.mode == "E" or args.mode == "evaluate":
        print("The mode selected will evaluate the model on a set of test "
              "data and save the history plots.")
        evaluation_print(model)
        plot_history_save(args.modelfile)

    elif args.mode == "P" or args.mode == "predict":
        print("The mode selected will predict the genres of a given synopsis.")
        predict_genres_print(model, args.synopsis)

    else:
        print("The mode selected does not exist. It must be one of the "
              "following:\n"
              "\tevaluate | E = evaluate the model and save history plots.\n"
              "\tpredict | P = use the model to predict genres of a given "
              "synopsis.\n Closing the application.")
        exit(0)