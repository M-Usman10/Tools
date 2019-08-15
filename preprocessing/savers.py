import os

import cv2
import numpy as np


def save_imgs(imgs, outDir, names=None):
    if names is None:
        names = np.array(range(len(imgs)))
        names = np.core.defchararray.add(names, '.jpg')
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    for idx, img in enumerate(imgs):
        cv2.imwrite(img, os.path.join(outDir, names[idx]))
