import numpy as np

def normalise_data(dataset, X_train, X_test):

    if dataset == 'cifar10' or dataset == 'cifar100':

        n_channels = X_train.shape[-1]

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # Normalisation
        X = np.vstack((X_train, X_test))

        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    elif dataset == 'mnist':

        # input image dimensions
        img_rows, img_cols = 28, 28

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    else:
        exit('Abort! Add normalisation for your own dataset')

    return X_train, X_test
