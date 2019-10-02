import pylab as plt
import numpy as np
#import seaborn as sns; sns.set()

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

## PCA

mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
Zpca = np.dot(x_train - mu, V.transpose())

Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

## Autoencoder

m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=(784,)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(x_train, x_train, batch_size=128, epochs=5, verbose=2,
                validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)  # bottleneck representation
Renc = m.predict(x_train)        # reconstruction


## plot

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('PCA')
plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(122)
plt.title('Autoencoder')
plt.scatter(Zenc[:5000,0], Zenc[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.tight_layout()

fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

def plot_pca_enc(Zpca, Zenc, y_train):

    idx = np.random.choice(x_train.shape[0], 5000)

    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
    ax0.set_title('PCA')
    ax0.scatter(Zpca[idx, 0], Zpca[idx, 1], c=y_train[idx], s=8, cmap=plt.cm.get_cmap('tab10'))
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_title('Autoencoder')
    im = ax1.scatter(Zenc[idx, 0], Zenc[idx, 1], c=y_train[idx], s=8, cmap=plt.cm.get_cmap('tab10'))
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig1.colorbar(im, ax=ax1)

    fig1.savefig('autoencoder_pca.png')

    return fig1

plot_pca_enc(Zpca, Zenc, y_train)
