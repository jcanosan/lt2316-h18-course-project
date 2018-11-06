# train.py
# This is the script to train either an LSTM or a CNN model for text
# classification for the AllMoviesDetailsCleaned.csv dataset.


from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pickle
import os
import time
from random import shuffle
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.layers import Input, Activation, LSTM, Embedding, Dropout, Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, Callback


def chunks(l, n):
    """
    Yield successive n-sized chunks from a list. Based on the function in:
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_tokenize_split():
    """
    Loads the data from the csv file, takes the needed columns and builds two
    parallel lists with them.
    Tokenizes, filters punctuation and one-word and empty sentences and splits
    into chunks of 50 words with their respective genres in a tuple
    ([chunk], [genres]).
    Randomizes the list, builds train, validation and test splits and saves them
    and a sorted list with all the unique genres into a pickle file.
    """
    csv_file = pd.read_csv('dataset/AllMoviesDetailsCleaned.csv', sep=';')
    # Choose the genres and synopses columns, drop rows with at least one empty
    # value (NaN)
    genres_synopses = pd.concat([csv_file["genres"], csv_file["overview"]],
                                axis=1)
    genres_synopses = genres_synopses.dropna()
    # The synopsis with less than 6 words normally have non-relevant content
    # like: "no overview", "not available", "third movie in series"
    genres_synopses = genres_synopses[genres_synopses['overview'].apply(
        lambda x: len(x.split(' ')) > 5)]

    # Split into two parallel lists
    genres = [gens.split('|') for gens in genres_synopses["genres"]]
    synopses = list(genres_synopses["overview"])

    # Tokenize
    synopses2 = []
    for syn in synopses:
        synopses2.append(
            [word_tokenize(sent) for sent in sent_tokenize(syn.lower())])

    # Filter punctuation
    synopses_token = []
    for syn, gen in zip(synopses2, genres):
        for sent in syn:
            synopses_token.append((
                list(filter(lambda word: word not in
                            '!"#$%&(--)*+,``\'\'-./:;<=>?@[\]^_{|}~...', sent)),
                gen))

    # Split sentences into chunks of 50 words and filter the one-word and empty
    synopses_gen_50 = []
    for sent, gen in synopses_token:
        if len(sent) > 1:
            if len(sent) > 50:
                for chunk in chunks(sent, 50):
                    synopses_gen_50.append((chunk, gen))
            else:
                synopses_gen_50.append((sent, gen))

    # Shuffle the data and split into train, validation and test set
    shuffle(synopses_gen_50)
    data30_len = int(len(synopses_gen_50) * 0.3)
    data50_len = int(len(synopses_gen_50) * 0.5)
    data70_len = int(len(synopses_gen_50) * 0.7)
    data80_len = int(len(synopses_gen_50) * 0.8)
    train30 = synopses_gen_50[:data30_len]
    train50 = synopses_gen_50[:data50_len]
    train70 = synopses_gen_50[:data70_len]
    validation10 = synopses_gen_50[data70_len:data80_len]
    test20 = synopses_gen_50[data80_len:]

    # Build a sorted list with all the unique genres
    all_genres = sorted(list(set(x for l in genres for x in l)))

    # Save the splits and the genres into pickle files
    save_file('splits/train30.pickle', train30)
    save_file('splits/train50.pickle', train50)
    save_file('splits/train70.pickle', train70)
    save_file('splits/validation10.pickle', validation10)
    save_file('splits/test20.pickle', test20)
    save_file('splits/all_genres.pickle', all_genres)


def save_file(filepath, data_to_save):
    """
    Saves a pickle file.

    :param filepath: the path of the file to save.
    :param data_to_save: the data to save.
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filepath):
    """
    Loads a pickle file.

    :param filepath: the path of the file to load.
    """
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


def prepare_input_data(syn_gen):
    """
    Prepares the input data for training a neural network model.

    Splits the input list into two parallel lists.
    Prepares the tokenizer and sequence and pad the synopsis list.

    :param syn_gen: A list of ([synopsis], [genres]).
    :return: the padded sequences of all the synopses and the word index (needed
    for the embedding layer of the CNN model).
    """
    # Extracts synopses
    syns = []
    for syn, gen in syn_gen:
        syns.append(syn)

    # If there is a tokenizer index saved into a file, load it.
    if os.path.isfile('tokenizer/tokenizer.pickle'):
        print("\tThere is already a tokenizer index inside tokenizer/. "
              "Loading...")
        tokenizer = load_file('tokenizer/tokenizer.pickle')
        print("\tDone!")
    # Else: prepare the tokenizer.
    else:
        print("\tBuilding a tokenizer index and saving into tokenizer/ "
              "folder...")
        tokenizer = Tokenizer(num_words=50000)
        tokenizer.fit_on_texts(syns)
        save_file('tokenizer/tokenizer.pickle', tokenizer)
        print("\tDone!")

    # Build and pad the sequences
    print("\tBuilding the sequences and pad for the input...")
    syns_seq = tokenizer.texts_to_sequences(syns)
    syns_pad = pad_sequences(syns_seq, maxlen=50)
    print("\tSequences and pad done!")

    return syns_pad, tokenizer.word_index


def prepare_output_categories(syn_gen, all_genres):
    """
    Transforms the genres list into vectors of 1 or 0, meaning that the genre is
    or is not in that synopsis, respectively.
    
    :param syn_gen: a list of ([synopsis], [genres]).
    :param all_genres: a list with all the unique genres.
    :return: the vector of genres for each synopsis.
    """

    print("\tBuilding categories vectors of 1 and 0 for the output...")
    # Extract genres
    gens = []
    for syn, gen in syn_gen:
        gens.append(gen)

    gens_vec = []
    for syn_gens in gens:
        syn_cats_vectors = []
        for gen in all_genres:
            if gen in syn_gens:
                syn_cats_vectors.append(1)
            else:
                syn_cats_vectors.append(0)
        gens_vec.append(np.array(syn_cats_vectors))
    gens_vec = np.array(gens_vec)
    print("\tCategories vectors done.")

    return gens_vec


class TimeHistory(Callback):
    """
    Calculates the time per epoch when training. Based on the class in:
    https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train_lstm_model(input1, output1, val_input1, val_output1):
    """
    Trains an LSTM Recurrent Neural Network model for text classification.

    :param input1: the padded sequences of all the synopses on the training set.
    :param output1: the vector of genres of all the synopses on the training
    set.
    :param val_input1: the padded sequences of all the synopses on the training
    set.
    :param val_output1: the vector of genres of all the synopses on the training
    set.
    :return: the trained LSTM model and the history of loss and accuracy.
    """

    input_layer = Input(shape=(input1.shape[1],))
    emb_layer = Embedding(50000, 100, input_length=input1.shape[1])(input_layer)
    lstm1 = LSTM(100)(emb_layer)
    dropout_layer = Dropout(0.2)(lstm1)
    dense_layer = Dense(output1.shape[1])(dropout_layer)
    cats_sigmoid_layer = Activation('sigmoid')(dense_layer)

    model = Model(inputs=[input_layer], outputs=[cats_sigmoid_layer])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=["accuracy"])

    # Checkpoint
    filepath_loss = "/scratch/guscanojo/rnn-genres_loss-{epoch:02d}-" \
                    "{loss:.2f}.hdf5"
    loss_checkpoint = ModelCheckpoint(filepath_loss, monitor='loss', verbose=1,
                                     save_best_only=True, mode='min')
    filepath_acc = "/scratch/guscanojo/rnn-genres_acc-{epoch:02d}-" \
                   "{acc:.2f}.hdf5"
    acc_checkpoint = ModelCheckpoint(filepath_acc, monitor='acc', verbose=1,
                                     save_best_only=True, mode='max')
    val_filepath_loss = "/scratch/guscanojo/val-rnn-genres_loss-{epoch:02d}-" \
                        "{val_loss:.2f}.hdf5"
    val_loss_checkpoint = ModelCheckpoint(val_filepath_loss, monitor='val_loss',
                                          verbose=1, save_best_only=True,
                                          mode='min')
    val_filepath_acc = "/scratch/guscanojo/val-rnn-genres_acc-{epoch:02d}-" \
                       "{val_acc:.2f}.hdf5"
    val_acc_checkpoint = ModelCheckpoint(val_filepath_acc, monitor='val_acc',
                                         verbose=1, save_best_only=True,
                                         mode='max')
    time_callback = TimeHistory()
    callbacks_list = [loss_checkpoint, acc_checkpoint,
                      val_loss_checkpoint, val_acc_checkpoint, time_callback]

    # Train and create the history of loss and acc
    model_history = model.fit([input1], [output1], epochs=50, batch_size=100,
                              callbacks=callbacks_list,
                              validation_data=(val_input1, val_output1))
    print("Time to train each epoch:\n", time_callback.times)

    return model, model_history


def glove_embedding_layer(word_index, pad_length):
    """
    Uses the model GloVe 6B vector 100d to obtain the vector representations for
    the words and prepares the embedding layer to train the convolutional neural
    network model.

    :param word_index: the index of words of the training dataset.
    :param pad_length: the length of the pads of the model input.
    :return: the embedding layer.
    """
    glove_dir = "glovedata/"
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.random.random((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, 100,
                                weights=[embedding_matrix],
                                input_length=pad_length,
                                trainable=False)

    return embedding_layer


def train_cnn_model(input1, output1, val_input1, val_output1, word_index):
    """
    Trains a Convolutional Neural Network model for text classification.

    :param input1: the padded sequences of all the synopses on the training set.
    :param output1: the vector of genres of all the synopses on the training
    set.
    :param val_input1: the padded sequences of all the synopses on the training
    set.
    :param val_output1: the vector of genres of all the synopses on the training
    set.
    :param word_index: the index of words of the training dataset to prepare the
    embedding layer.
    :return: the trained CNN model and the history of loss and accuracy.
    """

    embedding_layer = glove_embedding_layer(word_index, input1.shape[1])

    # Architecture 1
    input_layer = Input(shape=(input1.shape[1],))
    emb_layer = embedding_layer(input_layer)
    cov1_layer = Conv1D(128, 3, activation='relu')(emb_layer)
    pool1_layer = MaxPooling1D(3)(cov1_layer)
    cov2_layer = Conv1D(128, 3, activation='relu')(pool1_layer)
    pool2_layer = MaxPooling1D(3)(cov2_layer)
    dropout_layer = Dropout(0.2)(pool2_layer)
    flat_layer = Flatten()(dropout_layer)
    dense1_layer = Dense(128, activation='relu')(flat_layer)
    dense2_layer = Dense(output1.shape[1])(dense1_layer)
    cats_sigmoid_layer = Activation('sigmoid')(dense2_layer)

    # Architecture 2
    # input_layer = Input(shape=(input1.shape[1],))
    # emb_layer = embedding_layer(input_layer)
    # cov1_layer = Conv1D(256, 3, activation='relu')(emb_layer)
    # cov2_layer = Conv1D(256, 3, activation='relu')(cov1_layer)
    # pool1_layer = MaxPooling1D(3)(cov2_layer)
    # cov3_layer = Conv1D(256, 3, activation='relu')(pool1_layer)
    # cov4_layer = Conv1D(256, 3, activation='relu')(cov3_layer)
    # pool2_layer = MaxPooling1D(3)(cov4_layer)
    # dropout1_layer = Dropout(0.2)(pool2_layer)
    # flat_layer = Flatten()(dropout1_layer)
    # dense1_layer = Dense(128, activation='relu')(flat_layer)
    # dropout2_layer = Dropout(0.2)(dense1_layer)
    # dense2_layer = Dense(output1.shape[1])(dropout2_layer)
    # cats_sigmoid_layer = Activation('sigmoid')(dense2_layer)

    model = Model(inputs=[input_layer], outputs=[cats_sigmoid_layer])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=["accuracy"])

    # Checkpoint
    filepath_loss = "/scratch/guscanojo/cnn-genres_loss-{epoch:02d}-" \
                    "{loss:.2f}.hdf5"
    loss_checkpoint = ModelCheckpoint(filepath_loss, monitor='loss', verbose=1,
                                      save_best_only=True, mode='min')
    filepath_acc = "/scratch/guscanojo/cnn-genres_acc-{epoch:02d}-" \
                   "{acc:.2f}.hdf5"
    acc_checkpoint = ModelCheckpoint(filepath_acc, monitor='acc', verbose=1,
                                     save_best_only=True, mode='max')
    val_filepath_loss = "/scratch/guscanojo/val-cnn-genres_loss-{epoch:02d}-" \
                        "{val_loss:.2f}.hdf5"
    val_loss_checkpoint = ModelCheckpoint(val_filepath_loss, monitor='val_loss',
                                          verbose=1, save_best_only=True,
                                          mode='min')
    val_filepath_acc = "/scratch/guscanojo/val-cnn-genres_acc-{epoch:02d}-" \
                       "{val_acc:.2f}.hdf5"
    val_acc_checkpoint = ModelCheckpoint(val_filepath_acc, monitor='val_acc',
                                         verbose=1, save_best_only=True,
                                         mode='max')
    time_callback = TimeHistory()
    callbacks_list = [loss_checkpoint, acc_checkpoint,
                      val_loss_checkpoint, val_acc_checkpoint, time_callback]

    # Train
    model_history = model.fit([input1], [output1], epochs=50, batch_size=100,
                              callbacks=callbacks_list,
                              validation_data=(val_input1, val_output1))

    print("Time to train each epoch:\n", time_callback.times)

    return model, model_history


if __name__ == "__main__":

    # Arguments
    parser = ArgumentParser("python3 train.py")
    parser.add_argument('-M', '--model', type=str,
                        help="The type of model to train (REQUIRED):\n"
                             "\trnn = RNN (LSTM) model.\n"
                             "\tcnn = CNN model.",
                        required=True)
    parser.add_argument('trainingsize', type=str,
                        help="Name of the file to use for training the model, "
                             "corresponding to the size of the data. "
                             "It has to be one of the following:\n"
                             "\ttrain30.pickle: 30% of the dataset."
                             "\ttrain50.pickle: 50% of the dataset."
                             "\ttrain70.pickle: 70% of the dataset.")
    parser.add_argument('modelfile', type=str,
                        help="Path of the output model file.")
    args = parser.parse_args()

    if os.path.isfile('splits/train30.pickle') and \
            os.path.isfile('splits/train50.pickle') and \
            os.path.isfile('splits/train70.pickle') and \
            os.path.isfile('splits/validation10.pickle') and \
            os.path.isfile('splits/test20.pickle') and \
            os.path.isfile('splits/all_genres.pickle'):
        print("There are already a list of genres and train, test and "
              "validation splits inside the splits/ folder. Loading %s, "
              "validation10.pickle and all_genres.pickle."
              % args.trainingsize)
        train = load_file('splits/' + args.trainingsize)
        validation10 = load_file('splits/validation10.pickle')
        all_genres = load_file('splits/all_genres.pickle')

    else:
        print("Loading the dataset, preparing a list of genres and the train, "
              "test and validation splits...")
        load_tokenize_split()
        train = load_file('splits/' + args.trainingsize)
        validation10 = load_file('splits/validation10.pickle')
        all_genres = load_file('splits/all_genres.pickle')

    if args.model == "rnn":
        print("The option selected will train a RNN (LSTM) model.")
        print("Preparing the training data for the RNN model...")
        syns_pad = prepare_input_data(train)[0]
        gens_vec = prepare_output_categories(train, all_genres)
        print("Data prepared!")
        print("Preparing the validation data for the RNN model...")
        val_syns_pad = prepare_input_data(validation10)[0]
        val_gens_vec = prepare_output_categories(validation10, all_genres)
        print("Data prepared!")

        trained_model, trained_model_hist = \
            train_lstm_model(syns_pad, gens_vec, val_syns_pad, val_gens_vec)

    elif args.model == "cnn":
        print("The option selected will train a CNN model.")
        print("Preparing the training data for the CNN model...")
        syns_pad, word_index = prepare_input_data(train)
        gens_vec = prepare_output_categories(train, all_genres)
        print("Data prepared!")
        print("Preparing the validation data for the CNN model...")
        val_syns_pad = prepare_input_data(validation10)[0]
        val_gens_vec = prepare_output_categories(validation10, all_genres)
        print("Data prepared!")

        trained_model, trained_model_hist = \
            train_cnn_model(syns_pad, gens_vec, val_syns_pad, val_gens_vec,
                            word_index)

    else:
        print("The model option selected does not exist. It must be:\n"
              "\trnn = RNN (LSTM) model.\n"
              "\tcnn = CNN model.")
        exit(0)

    # Save the model and the respective history dict (for plot in test.py) into
    # a file
    trained_model.save("models/" + args.modelfile)
    print("Model saved in the path: models/%s" % args.modelfile)
    history_path = "models/" + args.modelfile.replace(".h5", "-hist.pickle")
    save_file(history_path, trained_model_hist.history)
    print("History dictionary saved in the path: %s" % history_path)