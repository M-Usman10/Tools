import keras
from ..Augment.bbox_util import *
from ..Augment.data_aug import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=32, shuffle=True,prob=0.3):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert (len(X)==len(y))
        self.X=X
        self.y=y
        self.on_epoch_end()
        self.prob=self.prob
        #RandomHSV(40, 40, 30)
        self.aug=Sequence([RandomHorizontalFlip(), RandomScale(diff = True), RandomTranslate(), RandomRotate(10), RandomShear()],probs=self.prob)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y
    def augment(self,img,box):
        img,box=self.aug(img,box)
        if random.random() < self.prob+0.2:
            img=keras.preprocessing.image.apply_brightness_shift(img,np.random.uniform(0.2,0.7))
        return img,box

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        selected_x,selected_y=self.X[list_IDs_temp], self.y[list_IDs_temp]
        for i in range(len(selected_y)):
            selected_x[i],selected_y[i]=self.augment(selected_x[i],selected_y[i])
        return selected_x,selected_y
