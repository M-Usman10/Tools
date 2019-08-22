import glob
import os
import xml.etree.ElementTree as ET

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


def read_given_images_with_details(root, names, size, total=None, preprocess=None):
    """
    Read images specified in names in given order
    :param root: Root directory of images
    :param names: list of image names placed in root
    :param preprocess: Functor for preprocessing
    :return: Numpy array of images
    """
    width_ratios = []
    height_ratios = []
    images = []
    if total is None:
        total = len(names)
    if preprocess is None:
        for idx, path in enumerate(names[:total]):
            images.append(cv2.imread(os.path.join(root, path)))
            my_print(idx)
    else:
        for idx, path in enumerate(names[:total]):
            img = cv2.imread(os.path.join(root, path))
            width_ratios.append(size[1] / img.shape[1])
            height_ratios.append(size[0] / img.shape[0])
            images.append(preprocess(img))
            my_print(idx)
    return np.array(images), height_ratios, width_ratios



def load_bbox_annotations(Path, names=None):
    if names is None:
        xmls = glob.glob(Path + "/*.xml")
    else:
        xmls = names
    AllBoxes = []
    for xml in xmls:
        root = ET.parse(xml).getroot()
        objects = root.findall('object')
        boxes = []
        for obj in objects:
            box = []
            for child in obj.find('bndbox'):
                box.append(int(child.text))
            new_box = [box[1], box[0], box[3], box[2]]
            boxes.append(new_box)
        AllBoxes.append(boxes)
    return AllBoxes
