import glob
import os

import cv2
import h5py
import numpy as np


def read_hdf5(inputPath, dataset="images"):
    files = h5py.File(inputPath, 'r')
    return files[dataset][:]


def read_images(inputPath, preprocess=None, format=''):
    """

    :param inputPath: Path of directory where images are placed
    :param preprocess: Functor for preprocessing
    :param format: '.jpg' etc.
    :return: Numpy array of images
    """
    images = []
    image_paths = glob.glob(os.path.join(inputPath, '*' + format))
    if preprocess is None:
        for path in image_paths:
            images.append(cv2.imread(path))
    else:
        for path in image_paths:
            images.append(preprocess(cv2.imread(path)))
    return np.array(images)
