from scipy import signal
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
from keras import optimizers
from matplotlib import pyplot as plt
from colour import Color


#########################################################################################
#                               Toy Convnet                                             #
#########################################################################################

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}


def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.matrix(np.ones(size)) * 1 / 3 for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        for _ in range(iterations):
            # draw a random batch for training:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives:
            p = np.dot(x, w[func_id].T).T
            e = p - y  # y - p
            loss = np.sum(np.multiply(e, e)) / batch_size + lamb / 2 * np.linalg.norm(w[func_id]) ** 2
            dl_dw = 2 / batch_size * np.dot(e, x) + lamb * w[func_id]

            # data for testing:
            x, y = X['test'], Y[func_id]['test']

            p = np.dot(x, w[func_id].T)
            e = y - p.T
            iteration_test_loss = np.sum(np.multiply(e, e)) / len(X['test']) + lamb / 2 * np.linalg.norm(
                w[func_id]) ** 2

            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

        w[func_id] = w[func_id] / np.linalg.norm(w[func_id])

    loss = {'train': training_loss,
            'test': test_loss}

    return w, loss


def forward(toy_model, x):
    """
    Given the Toy Convnet model, fill up a dictionary with the forward pass values.
    :param toy_model: the model
    :param x: the input of the Toy Concnet model
    :return: a dictionary with the forward pass values
    """

    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)),
                           signal.convolve2d(x, [np.array(toy_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)),
                           signal.convolve2d(x, [toy_model['w2']], mode='same'))
    fwd['m'] = np.maximum(fwd['o1'], fwd['o2'])
    fwd['m_argmax'] = np.where(fwd['o1'] >= fwd['o2'], 0, 1)
    fwd['p'] = np.dot(fwd['m'], toy_model['u'])

    return fwd


def backprop(toy_model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param toy_model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """

    mat_0 = np.matrix(np.zeros(batch_size))

    # dl_dw1
    dl_dp = 2 * (fwd['p'] - y)
    dp_dm = np.matrix(toy_model['u']).T

    dm_do1 = np.array([np.where(fwd['o1'][:, 0] >= fwd['o1'][:, 1], 1, 0),
                       np.where(fwd['o1'][:, 1] >= fwd['o1'][:, 0], 1, 0),
                       np.where(fwd['o1'][:, 2] >= fwd['o1'][:, 3], 1, 0),
                       np.where(fwd['o1'][:, 3] >= fwd['o1'][:, 2], 1, 0)])

    do1_dw1 = np.array([np.vstack((mat_0, np.multiply(fwd['x'][:, 0:2].T,
                                                      np.where(fwd['o1'][:, 0] > 0, 1, 0)))).T,
                        np.multiply(fwd['x'][:, 0:3].T, np.where(fwd['o1'][:, 1] > 0, 1, 0)).T,
                        np.multiply(fwd['x'][:, 1:4].T, np.where(fwd['o1'][:, 2] > 0, 1, 0)).T,
                        np.vstack((np.multiply(fwd['x'][:, 2:4].T,
                                               np.where(fwd['o1'][:, 3] > 0, 1, 0)), mat_0)).T
                        ])  # shape (4,batch_size,3)

    dl_do1 = np.multiply(np.multiply(dl_dp, dp_dm), dm_do1)
    dl_dw1 = np.tensordot(dl_do1, do1_dw1, axes=2)

    # dl_dw2
    dm_do2 = np.array([np.where(fwd['o2'][:, 0] >= fwd['o2'][:, 1], 1, 0),
                       np.where(fwd['o2'][:, 1] >= fwd['o2'][:, 0], 1, 0),
                       np.where(fwd['o2'][:, 2] >= fwd['o2'][:, 3], 1, 0),
                       np.where(fwd['o2'][:, 3] >= fwd['o2'][:, 2], 1, 0)])

    do2_dw2 = np.array([np.vstack((mat_0, np.multiply(fwd['x'][:, 0:2].T,
                                                      np.where(fwd['o2'][:, 0] > 0, 1, 0)))).T,
                        np.multiply(fwd['x'][:, 0:3].T, np.where(fwd['o2'][:, 1] > 0, 1, 0)).T,
                        np.multiply(fwd['x'][:, 1:4].T, np.where(fwd['o2'][:, 2] > 0, 1, 0)).T,
                        np.vstack((np.multiply(fwd['x'][:, 2:4].T,
                                               np.where(fwd['o2'][:, 3] > 0, 1, 0)), mat_0)).T
                        ])  # shape (4,batch_size,3)

    dl_do2 = np.multiply(np.multiply(dl_dp, dp_dm), dm_do2)

    dl_dw2 = np.tensordot(dl_do2, do2_dw2, axes=2)

    # dl_du
    dl_du = np.dot(dl_dp.T, fwd['m']) / batch_size

    dl_dw1 = dl_dw1 / np.linalg.norm(dl_dw1)
    dl_dw2 = dl_dw2 / np.linalg.norm(dl_dw2)
    dl_du = dl_du / np.linalg.norm(dl_du)

    return dl_dw1, dl_dw2, dl_du


def learn_toy(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn the Toy Convnet model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.array([-0.2, 0.2, 0.99])
        models[func_id]['w2'] = np.ones(3)
        models[func_id]['u'] = np.ones(4)

        # train the network:
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = np.sum((fwd['p'] - y) ** 2) / batch_size + \
                   lamb / 2 * (np.linalg.norm(models[func_id]['w1']) ** 2 +
                               np.linalg.norm(models[func_id]['w2']) ** 2)
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = np.sum((test_fwd['p'] - Y[func_id]['test']) ** 2) * \
                                  1 / X['test'].shape[0] + \
                                  lamb / 2 * (np.linalg.norm(models[func_id]['w1']) ** 2 +
                                              np.linalg.norm(models[func_id]['w2']) ** 2)

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    loss = {'train': training_loss,
            'test': test_loss}

    return models, loss


def plot_comparison(linear_loss, toy_loss):
    """
    plot the learning curves of the linear model and the toy convnet model
    :param linear_loss: training history of linear model
    :param toy_loss: training history of toy convnet model
    :return: the plot
    """

    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))

    red = Color("red")
    colors = list(red.range_to(Color("green"), num_of_functions))

    for func_id in range(0, num_of_functions):
        col = colors[func_id]
        ax0.plot(linear_loss['train'][func_id], 'g--', c=str(col), linewidth=2,
                 label='fct ' + str(func_id) + ' training loss')
        ax0.plot(linear_loss['test'][func_id], 'g-', c=str(col), linewidth=2,
                 label='fct. ' + str(func_id) + ' test loss')

    ax0.legend(loc='upper right')
    ax0.set_title('linear model')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')

    for func_id in range(0, num_of_functions):
        col = colors[func_id]
        ax1.plot(toy_loss['train'][func_id], 'g--', c=str(col), linewidth=2,
                 label='fct. ' + str(func_id) + ' training loss')
        ax1.plot(toy_loss['test'][func_id], 'g-', c=str(col), linewidth=2,
                 label='fct. ' + str(func_id) + ' test loss')

    ax1.legend(loc='upper right')
    ax1.set_title('Toy Convnet')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    fig1.savefig('linear_toynet.png')

    return fig1


########################################################################################
#                              MNIST Classifier                                        #
########################################################################################

def linear(mnist_data, batch_size, epochs):
    """
    defines and trains a linear model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :return: the trained linear model and its training history
    """

    # load and pre-process MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # define model
    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(784,)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    # train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))

    return model, history

def mlp(mnist_data, batch_size, epochs, dropout_rate):
    """
    defines and trains a mlp model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param dropout_rate: dropout rate between the layers
    :return: the trained mlp model and its training history
    """

    # load and pre-process MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # define model
    model = Sequential()
    model.add(Dense(512, activation='elu', input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0000037),
                  metrics=['accuracy'])

    # train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))

    return model, history


def cnn(mnist_data, batch_size, epochs, dropout_rate):
    """
    defines and trains a cnn model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param dropout_rate: dropout rate between the layers
    :return: the trained cnn model and its training history
    """

    # load and pre-process MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist_data

    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # define model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizers.Adam(lr=0.0000037),   # default learning rate = 0.001
                      metrics=['accuracy'])

    # train model
    history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(x_test, y_test))

    return model, history


def plot_linear_mlp_cnn(linear_history, mlp_history, cnn_history):
    """
    creates a plot to compare the learning curve of the three models
    :param linear_history: training history of linear model
    :param mlp_history: training history of mlp model
    :param cnn_history: training history of cnn model
    :return: the plot
    """
    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))

    # accuracy
    ax0.plot(linear_history.history['acc'], 'b--', linewidth=2,
             label='linear train')
    ax0.plot(linear_history.history['val_acc'], 'b-', linewidth=2,
             label='linear test')

    ax0.plot(mlp_history.history['acc'], 'g--', linewidth=2,
             label='mlp train')
    ax0.plot(mlp_history.history['val_acc'], 'g-', linewidth=2,
             label='mlp test')

    ax0.plot(cnn_history.history['acc'], 'y--', linewidth=2,
             label='cnn train')
    ax0.plot(cnn_history.history['val_acc'], 'y-', linewidth=2,
             label='cnn test')

    ax0.legend(loc='lower right')
    ax0.set_title('model accuracy')
    ax0.set_ylabel('accuracy')
    ax0.set_xlabel('epoch')

    # loss
    ax1.plot(linear_history.history['loss'], 'b--', linewidth=2,
             label='linear train')
    ax1.plot(linear_history.history['val_loss'], 'b-', linewidth=2,
             label='linear test')

    ax1.plot(mlp_history.history['loss'], 'g--', linewidth=2,
             label='mlp train')
    ax1.plot(mlp_history.history['val_loss'], 'g-', linewidth=2,
             label='mlp test')

    ax1.plot(cnn_history.history['loss'], 'y--', linewidth=2,
             label='cnn train')
    ax1.plot(cnn_history.history['val_loss'], 'y-', linewidth=2,
             label='cnn test')

    ax1.legend(loc='upper right')
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')

    fig1.savefig('linear_mlp_cnn.png')

    return fig1


#########################################################################################
#                               Hyperparameter                                          #
#########################################################################################
def mlp_hyper(dropout_rate, learning_rate):
    """
    defines the mlp model that is used for hyper parameter training
    :param dropout_rate: dropout rate between the layers
    :param learning_rate: learning rate of optimization - here: SGD
    :return: the compiled mlp model
    """
    
    # define optimizer with specific learning rate
    adam = optimizers.Adam(lr=learning_rate)  # default learning rate = 0.001

    model = Sequential()
    model.add(Dense(512, activation='elu', input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def hyperparameter(mnist_data, dropout_rate, batch_size, epochs, learning_rate_list):
    """
    Given a list of hyperparameters trains a mlp model for those parameters
    :param mnist_data: the mnist data
    :param dropout_rate: dropout rate in between the cnn layers
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param learning_rate_list: list of learning rates for hyper parameter training
    :return: learning_rate_list: list of learning rates
             learning_curve_list: learning histories of mlp models
    """

    # load and pre-process MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # iterate over different learning rates
    learning_curve_list = []
    for i in range(0, len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        model = mlp_hyper(dropout_rate, learning_rate)  # create model with learning_rate

        print('--------------- iteration ' + str(i) + '  --------------------')
        print('learning rate:  ' + str(learning_rate) + '  --------------------')

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(x_test, y_test))

        learning_curve = history.history
        print('score:  ' + str(history.history))
        
        # save training history for each learning_rate
        learning_curve_list.append(learning_curve)
        del model

    return learning_rate_list, learning_curve_list


def plot_hyper(hyper_scores, hyper_lrate):
    """
    plots the learning curve for the different hyper parameters (learning_rate)
    :param hyper_scores: training history for different learning rates
    :param hyper_lrate: list of learning_rates
    :return: the plot
    """

    fig1, (ax0) = plt.subplots(ncols=1, figsize=(8, 5))

    red = Color("blue")
    colors = list(red.range_to(Color("brown"), len(hyper_scores)))

    for i in range(0, len(hyper_scores)):
        col = colors[i]
        ax0.plot(hyper_scores[i]['loss'], '--', c=str(col), linewidth=2)
        ax0.plot(hyper_scores[i]['val_loss'], '-', c=str(col), linewidth=2,
                 label= 'learning_rate = ' + str(hyper_lrate[i].round(7)))

    ax0.legend(loc='upper right')
    ax0.set_title('Learning Curve MLP')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')

    fig1.savefig('hyper_fine.png')
    return fig1

#########################################################################################
#                                 Autoencoder                                           #
#########################################################################################

def autoencoder(mnist_data, batch_size, epochs):
    """
    trains an Autoencoder and PCA to reduce dimensionality of data and plots results
    :param mnist_data: data to be reduced in dimensionality
    :param batch_size: batch size for training of Autoencoder
    :param epochs: epochs for training of Autoencoder
    :return: fig1: learning curve of Autoencoder
             fig2: comparison PCA vs. Autoencoder
             history: training history of Autoencoder
             auto: trained Autoencoder model
    """

    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255

    # dimensionality reduction with PCA
    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())  # encoded data

    # dimensionality reduction with Autoencoder
    auto = Sequential()
    auto.add(Dense(512, activation='elu', input_shape=(784,)))
    auto.add(Dense(128, activation='elu'))
    auto.add(Dense(2, activation='linear', name="2D"))
    auto.add(Dense(128, activation='elu'))
    auto.add(Dense(512, activation='elu'))
    auto.add(Dense(784, activation='sigmoid'))
    auto.compile(loss='mean_squared_error', optimizer='Adam')
    history = auto.fit(x_train, x_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2,
                       validation_data=(x_test, x_test))

    encoder = Model(auto.input, auto.get_layer('2D').output)
    Zenc = encoder.predict(x_train)  # encoded data

    # plot learning function of Autoencoder
    fig1, (ax0) = plt.subplots(ncols=1, figsize=(8, 5))

    ax0.plot(history.history['loss'], 'b--', linewidth=2, label='training loss')
    ax0.plot(history.history['val_loss'], 'b-', linewidth=2, label='test loss')

    ax0.legend(loc='upper right')
    ax0.set_title('Learning Curve')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')

    fig1.savefig('autoencoder_learning.png')

    # plot comparison PCA - Autoencoder
    idx = np.random.choice(x_train.shape[0], 5000)  # 5000 random values
    fig2, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))

    ax0.set_title('PCA')
    ax0.scatter(Zpca[idx, 0], Zpca[idx, 1], c=y_train[idx], s=8,
                cmap=plt.cm.get_cmap('tab10'))
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_title('Autoencoder')
    im = ax1.scatter(Zenc[idx, 0], Zenc[idx, 1], c=y_train[idx], s=8,
                     cmap=plt.cm.get_cmap('tab10'))
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig2.colorbar(im, ax=ax1)

    fig2.savefig('autoencoder_pca.png')

    return fig1, fig2, history, auto


if __name__ == '__main__':

    #########################################################################################
    #                                  Toy Convnet                                          #
    #########################################################################################

    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((10000, size)) - .5),
             test=5 * (np.random.random((2000, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
                1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
                1 + np.random.randn(X['test'].shape[0]) * .01)}
        for i in range(len(y))}

    # train the linear model and the toy convnet model
    batch_size = 1000
    w, linear_loss = learn_linear(X=X, Y=Y, batch_size=batch_size, lamb=0.2,
                                  iterations=50, learning_rate=0.01)
    models, toy_loss = learn_toy(X=X, Y=Y, batch_size=batch_size, lamb=0.2,
                                 iterations=200, learning_rate=0.01)

    # plot the learning curves of the models
    fig1 = plot_comparison(linear_loss, toy_loss)


    #########################################################################################
    #                               MNIST Classifier                                        #
    #########################################################################################

    # load MNIST data set
    mnist_data = mnist.load_data()

    batch_size = 128
    epochs = 20
    dropout_rate = 0.2

    # linear model
    print('---------- train linear model ----------')
    linear_model, linear_history = linear(mnist_data, batch_size, epochs)
    print('test loss: ' + str(linear_history.history['val_loss']))

    # mlp model
    print('---------- train mlp model ----------')
    mlp_model, mlp_history = mlp(mnist_data, batch_size, epochs, dropout_rate)
    print('test loss: ' + str(mlp_history.history['val_loss']))

    # cnn model
    print('---------- train cnn model ----------')
    cnn_model, cnn_history = cnn(mnist_data, batch_size, epochs, dropout_rate)
    print('test loss: ' + str(cnn_history.history['val_loss']))

    fig2 = plot_linear_mlp_cnn(linear_history, mlp_history, cnn_history)


    #########################################################################################
    #                               Hyperparameter                                          #
    #########################################################################################
    print('---------- tune hyper parameters ----------')

    mnist_data = mnist.load_data()

    batch_size = 128
    epochs = 10
    dropout_rate = 0.2

    # list of learning_rates for hyper-parameter tuning
    # learning_rate_list = [0.000001, 0.00001, 0.0001, 0.001, 0.006]  # coarse values
    learning_rate_list = np.linspace(0.000001, 0.0000064, 5)          # fine values
    hyper_lrate, hyper_scores = hyperparameter(mnist_data, dropout_rate, batch_size, epochs, learning_rate_list)

    fig3 = plot_hyper(hyper_scores, hyper_lrate)


    #########################################################################################
    #                                 Autoencoder                                           #
    #########################################################################################

    print('---------- autoencoder ----------')

    batch_size = 128
    epochs = 20
    fig4, fig5, auto_history, auto_model = autoencoder(mnist_data, batch_size, epochs)

    plt.show()