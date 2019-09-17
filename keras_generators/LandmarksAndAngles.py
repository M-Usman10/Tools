import keras

from ..Augment.data_aug import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, y, z, batch_size=32, shuffle=True, prob=0.3):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert (len(X) == len(y))
        self.X = X
        self.y = y
        self.z = z
        self.on_epoch_end()
        self.prob = prob
        # RandomHSV(40, 40, 30)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y, z = self.__data_generation(indexes)
        return X, y, z

    def augment(self, img):
        if random.random() < self.prob + 0.2:
            img = keras.preprocessing.image.apply_brightness_shift(img, np.random.uniform(0.2, 0.7))
        return img

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        selected_x, selected_y, selected_z = self.X[list_IDs_temp], self.y[list_IDs_temp], self.z[list_IDs_temp]
        for i in range(len(selected_y)):
            selected_x[i] = self.augment(selected_x[i])
        return selected_x, selected_y, selected_z
