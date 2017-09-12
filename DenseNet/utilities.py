import numpy as np
import keras.backend as K

def normalise_data(dataset, X_train, X_test):

    if dataset == 'cifar10' or dataset == 'cifar100':

        if K.image_dim_ordering() == "th":
            n_channels = X_train.shape[1]
        else:
            n_channels = X_train.shape[-1]

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # Normalisation
        X = np.vstack((X_train, X_test))
        # 2 cases depending on the image ordering
        if K.image_dim_ordering() == "th":
            for i in range(n_channels):
                mean = np.mean(X[:, i, :, :])
                std = np.std(X[:, i, :, :])
                X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
                X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

        elif K.image_dim_ordering() == "tf":
            for i in range(n_channels):
                mean = np.mean(X[:, :, :, i])
                std = np.std(X[:, :, :, i])
                X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
                X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    elif dataset == 'mnist':

        # input image dimensions
        img_rows, img_cols = 28, 28

        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    else:
        exit('Abort! Add normalisation for your own dataset')

    return X_train, X_test
