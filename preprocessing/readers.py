import glob
import os

import cv2
import h5py
import numpy as np

from ..others.print import my_print


def read_hdf5(inputPath, dataset="images"):
    files = h5py.File(inputPath, 'r')
    return files[dataset][:]


def read_images(inputPath, preprocess=None, format='', sorted=False):
    """

    :param inputPath: Path of directory where images are placed
    :param preprocess: Functor for preprocessing
    :param format: '.jpg' etc.
    :return: Numpy array of images
    """
    images = []
    image_paths = np.array(glob.glob(os.path.join(inputPath, '*' + format)))
    if sorted:
        l = []
        for path in image_paths:
            l.append(int(os.path.basename(path).split('.')[0]))
        idx = np.argsort(l)
        image_paths = image_paths[idx]
    if preprocess is None:
        for idx, path in enumerate(image_paths):
            images.append(cv2.imread(path))
            my_print(idx)
    else:
        for idx, path in enumerate(image_paths):
            images.append(preprocess(cv2.imread(path)))
            my_print(idx)
    return np.array(images)


def read_given_images(root, names, preprocess=None):
    """
    Read images specified in names in given order
    :param root: Root directory of images
    :param names: list of image names placed in root
    :param preprocess: Functor for preprocessing
    :return: Numpy array of images
    """
    images = []
    if preprocess is None:
        for idx, path in enumerate(names):
            images.append(cv2.imread(os.path.join(root, path)))
            my_print(idx)
    else:
        for idx, path in enumerate(names):
            images.append(preprocess(cv2.imread(os.path.join(root, path))))
            my_print(idx)
    return np.array(images)
