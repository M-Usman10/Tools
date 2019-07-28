import h5py
import numpy as np


def readHdf5(inputPath,dataset="images"):
    files = h5py.File(inputPath, 'r')
    return files[dataset][:]

