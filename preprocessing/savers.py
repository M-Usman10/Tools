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
        cv2.imwrite(os.path.join(outDir, names[idx]), img)


def save_boxes_txt(boxes,obj_names,img_name,outPath,valueSeparator=' ', objectSeparator='\n'):
    """

    Parameters
    ----------
    boxes
    obj_names
    img_names
    outPath
    valueSeparator
    objectSeparator

    Returns
    -------

    """
    import os
    if not (os.path.isdir(outPath)):
        os.mkdir(outPath)

    filename = os.path.join(outPath, os.path.basename(img_name).split('.')[0] + '.txt')
    with open(filename, 'w') as file:
        for i in range(len(boxes)):
            file.write(obj_names[i] + valueSeparator + valueSeparator.join(boxes[i].astype(str)) + objectSeparator)


def save_boxes_yolo(boxes,obj_names,outPath,dict_,imgPath,valueSeparator=',', objectSeparator=' '):
    with open(outPath, 'a') as file:
        file.write(imgPath+objectSeparator)
        for i in range(len(boxes)):
            if i < len(boxes)-1:
                file.write(valueSeparator.join(boxes[i].astype(str))+ valueSeparator +dict_[obj_names[i]]  +  objectSeparator)
            else:
                file.write(valueSeparator.join(boxes[i].astype(str)) + valueSeparator + dict_[obj_names[i]])

        file.write("\n")