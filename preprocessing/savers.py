import os

import cv2
import numpy as np


def save_images(imgs, outDir, names=None):
    if names is None:
        names = np.array(range(len(imgs))).astype(str)
        names = np.core.defchararray.add(names, np.array(['.jpg'] * len(imgs)))
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    for idx, img in enumerate(imgs):
        print(outDir, names[idx])
        print(os.path.join(outDir, names[idx]))
        cv2.imwrite(img, os.path.join(outDir, names[idx]))
