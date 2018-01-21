from keras import models
from keras import layers

import numpy as np


#############
# data prep #
#############

def vectorize_words(sequences, dimension):
    """
    One hot encode each example in sequences
    Returns a sparse matrix of shape
    (len(sequences), dimension)
    """

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        # sequence is a list which we use to
        # access a subset of the array results
        # at row i
        results[i, sequence] = 1.

    return results


########################################
# building and fitting neural networks #
########################################

def nn_architecture(num_act_units, output_act_units, input_unit_shape):
    """
    Custom neural network architecture for the reuters dataset
    Length of list in num_act_units correspond to number of
    hidden layers in network

    num_act_units (number of activation units): list
    output_act_units (number of activation units in output layer): int
    input_unit_shape (shape of input features): tuple
    """

    num_layers = len(num_act_units)

    custom_model = models.Sequential()

    # first hidden layer (input are the one hot encoded top 10,000 words)
    custom_model.add(layers.Dense(num_act_units[0], activation='relu', input_shape=input_unit_shape))

    # remaining num_layers - 1 hidden layers
    for l in range(num_layers - 1):
        custom_model.add(layers.Dense(num_act_units[l + 1], activation='relu'))

    # final layers (46 possible classes)
    custom_model.add(layers.Dense(output_act_units, activation='softmax'))

    custom_model.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    return custom_model


def nn_fit(x_train, y_train, x_val, y_val, model, num_epochs, batch_size):
    """
    Fit model using training data and validate
    Returns the history of the training session
    """

    history = model.fit(x_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=2)

    return history

