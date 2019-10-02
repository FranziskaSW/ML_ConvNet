import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import pickle

from keras import optimizers
from matplotlib import pyplot as plt
from colour import Color


def linear(mnist_data, batch_size, epochs):
    """
    defines and trains a linear model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :return: the trained linear model and its training history
    """
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(784,)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))

    return [model, history]

def mlp(mnist_data, batch_size, epochs, dropout_rate):
    """
    defines and trains a mlp model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param dropout_rate: dropout rate between the layers
    :return: the trained mlp model and its training history
    """
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model = Sequential()
    model.add(Dense(512, activation='elu', input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))

    return [model, history]


def cnn(mnist_data, batch_size, epochs, dropout_rate):
    """
    defines and trains a cnn model
    :param mnist_data: the MNIST data
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param dropout_rate: dropout rate between the layers
    :return: the trained cnn model and its training history
    """

    (x_train, y_train), (x_test, y_test) = mnist_data

    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

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
                      optimizer='Adam',   # default learning rate = 0.01
                      metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_test, y_test))

    return [model, history]

def cnn_hyper(input_shape, dropout_rate, learning_rate):
    """
    defines the cnn model that is used for hyper parameter training
    :param input_shape: shape of the input data (image)
    :param dropout_rate: dropout rate between the layers
    :param learning_rate: learning rate of optimization - here: adam
    :return: the compiled cnn model
    """
    #sgd = optimizers.sgd(lr=learning_rate)
    adam = optimizers.adam(lr=learning_rate) # default lr=0.001

    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(dropout_rate))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(dropout_rate))
    cnn_model.add(Dense(10, activation='softmax'))

    cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=adam,  # default learning rate = 0.01
                      metrics=['accuracy'])
    return cnn_model


def hyperparameter(mnist_data, dropout_rate, batch_size, epochs, learning_rate_list):
    """
    Given a list of hyperparameters trains a cnn model for those parameters
    :param mnist_data: the mnist data
    :param dropout_rate: dropout rate in between the cnn layers
    :param batch_size: batch size for training
    :param epochs: epochs of training
    :param learning_rate_list: list of learning rates for hyper parameter training
    :return: learning_rate_list (list of learning rates), learning_curve_list (learning history of cnn model)
    """
    (x_train, y_train), (x_test, y_test) = mnist_data

    x_train = x_train / np.max(x_train)
    x_test = x_test / np.max(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    learning_curve_list = []
    for i in range(0, len(learning_rate_list)):   # TODO: coarse: Hyperparameter random layout (random uniform in interval)
                                          #      fine  : grid around the maximum
        learning_rate = learning_rate_list[i]
        model = cnn_hyper(input_shape, dropout_rate, learning_rate)

        print('--------------- iteration ' + str(i) + '  --------------------')
        print('learning rate:  ' + str(learning_rate) + '  --------------------')

        history = model.fit(x_train, y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=2,
                               validation_data=(x_test, y_test))

        learning_curve = history.history
        print('score:  ' + str(history.history))
        learning_curve_list.append(learning_curve)
        del model

    return [learning_rate_list, learning_curve_list]


def plot_comparison(linear_history, mlp_history, cnn_history):
    """
    creates a plot to compare the learning curve of the three models
    :param linear_history: training history of linear model
    :param mlp_history: training history of mlp model
    :param cnn_history: training history of cnn model
    :return: the plot
    """
    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

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

    fig1.savefig('linear_mlp_cnn_adam.png')

    return fig1


def plot_hyper(hyper_scores, hyper_lrate):

    fig1, (ax0) = plt.subplots(ncols=1, figsize=(12, 6))

    red = Color("blue")
    colors = list(red.range_to(Color("orange"), len(hyper_scores)))

    for i in range(0,len(hyper_scores)):
        col = colors[i]
        ax0.plot(hyper_scores[i]['loss'], '--', c=str(col), linewidth=2)
        ax0.plot(hyper_scores[i]['val_loss'], '-', c=str(col), linewidth=2,
                         label= 'learning_rate = ' + str(hyper_lrate[i].round(6)))

    ax0.legend(loc='upper right')
    ax0.set_title('Training Curve CNN')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')

    fig1.savefig('hyper_adam_coarse.png')
    return fig1

def autoencoder(mnist_data, batch_size, epochs):
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255

    ## PCA

    mu = x_train.mean(axis=0)
    U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())

    ## Autoencoder

    auto = Sequential()
    auto.add(Dense(512, activation='elu', input_shape=(784,)))
    auto.add(Dense(128, activation='elu'))
    auto.add(Dense(2, activation='linear', name="bottleneck"))
    auto.add(Dense(128, activation='elu'))
    auto.add(Dense(512, activation='elu'))
    auto.add(Dense(784, activation='sigmoid'))
    auto.compile(loss='mean_squared_error', optimizer='Adam')
    history = auto.fit(x_train, x_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2,
                       validation_data=(x_test, x_test))

    encoder = Model(auto.input, auto.get_layer('bottleneck').output)
    Zenc = encoder.predict(x_train)  # bottleneck representation

    ## plot learning function

    fig1, (ax0) = plt.subplots(ncols=1, figsize=(12, 6))

    ax0.plot(history.history['loss'], 'b--', linewidth=2, label='training loss')
    ax0.plot(history.history['val_loss'], 'b-', linewidth=2, label='test loss')

    ax0.legend(loc='upper right')
    ax0.set_title('Learning Curve')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')

    fig1.savefig('autoencoder_learning.png')

    ## plot comparison PCA - Autoencoder

    idx = np.random.choice(x_train.shape[0], 5000)

    fig2, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
    ax0.set_title('PCA')
    ax0.scatter(Zpca[idx, 0], Zpca[idx, 1], c=y_train[idx], s=8, cmap=plt.cm.get_cmap('tab10'))
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_title('Autoencoder')
    im = ax1.scatter(Zenc[idx, 0], Zenc[idx, 1], c=y_train[idx], s=8, cmap=plt.cm.get_cmap('tab10'))
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig2.colorbar(im, ax=ax1)

    fig2.savefig('autoencoder_pca.png')

    return [fig1, fig2, history, auto]


if __name__ == '__main__':

    # load MNIST data set
    mnist_data = mnist.load_data()

    batch_size = 128
    epochs = 20
    dropout_rate = 0.2
    # learning rate default lr=0.01 (SGD)

    # linear model
    print('---------- train linear model ----------')
    linear_model, linear_history = linear(mnist_data, batch_size, epochs)
    print('test loss: ' + str(linear_history.history['val_loss']))

    linear_model.save('linear_model.h5')
    with open('linear_history.pickle', 'wb') as file_pi:
        pickle.dump(linear_history.history, file_pi)

    # mlp model
    print('---------- train mlp model ----------')
    mlp_model, mlp_history = mlp(mnist_data, batch_size, epochs, dropout_rate)
    print('test loss: '+ str(mlp_history.history['val_loss']))

    mlp_model.save('mlp_model.h5')
    with open('mlp_history.pickle', 'wb') as file_pi:
        pickle.dump(mlp_history.history, file_pi)

    # cnn model
    print('---------- train cnn model ----------')
    cnn_model, cnn_history = cnn(mnist_data, batch_size, epochs, dropout_rate)
    print('test loss: ' + str(cnn_history.history['val_loss']))

    cnn_model.save('cnn_model.h5')
    with open('cnn_history.pickle', 'wb') as file_pi:
        pickle.dump(cnn_history.history, file_pi)

    fig1 = plot_comparison(linear_history, mlp_history, cnn_history)

    ####################################################################################################
    #             hyper parameter tuning
    ####################################################################################################

    print('---------- tune hyper parameters ----------')

    mnist_data = mnist.load_data()

    batch_size = 128
    epochs = 8
    dropout_rate = 0.2

    learning_rate_list = np.linspace(0.001-0.0008, 0.001+0.0008, 3) # centered at default value
    hyper_lrate, hyper_scores = hyperparameter(mnist_data, dropout_rate, batch_size, epochs, learning_rate_list)

    with open('hyper_adam_coarse.pickle', 'wb') as file_pi:
        pickle.dump(hyper_lrate, file_pi)

    fig2 = plot_hyper(hyper_scores, hyper_lrate)

    ####################################################################################################
    #             autoencoder
    ####################################################################################################

    print('---------- autoencoder ----------')

    batch_size = 128
    epochs = 20
    fig3, fig4, auto_history, auto_model = autoencoder(mnist_data, batch_size, epochs)

    auto_model.save('autoencoder.h5')
    with open('auto_history.pickle', 'wb') as file_pi:
        pickle.dump(auto_history.history, file_pi)

    plt.show()
