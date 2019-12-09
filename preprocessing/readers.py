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


def read_images(inputPath, preprocess=None, format='', total=None, sorted=False):
    """

    :param inputPath: Path of directory where images are placed
    :param preprocess: Functor for preprocessing
    :param format: '.jpg' etc.
    :return: Numpy array of images
    """

    images = []
    image_paths = np.array(glob.glob(os.path.join(inputPath, '*' + format)))
    if total is None:
        total = len(image_paths)
    if sorted:
        l = []
        for path in image_paths:
            l.append(int(os.path.basename(path).split('.')[0]))
        idx = np.argsort(l)
        image_paths = image_paths[idx]
    if preprocess is None:
        for idx, path in enumerate(image_paths[:total]):
            images.append(cv2.imread(path))
            my_print("R   eading Img: " + str(idx + 1) + "/" + str(total))
    else:
        for idx, path in enumerate(image_paths[:total]):
            images.append(preprocess(cv2.imread(path)))
            my_print("Reading Img: " + str(idx + 1) + "/" + str(total))
    names = []
    if total is None:
        total=len(image_paths)
    for path in image_paths[:total]:
        names.append(os.path.basename(path))
    return np.array(images), np.array(names)


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


def read_given_images_with_ratios(root, names, size, total=None, preprocess=None):
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


def read_boxes_from_xml(path):
    """

    Parameters
    ----------
    path: Path of xml file

    Returns
    -------
        2D array of bounding boxes
    """

    root = ET.parse(path).getroot()
    objects = root.findall('object')
    boxes = []
    for obj in objects:
        box = []
        for child in obj.find('bndbox'):
            box.append(int(child.text))
        new_box = [box[1], box[0], box[3], box[2]]
        boxes.append(new_box)

def load_bbox_annotations(Path, names=None):
    if names is None:
        xmls = glob.glob(Path + "/*.xml")
    else:
        xmls = names
    AllBoxes = []
    for xml in xmls:
        AllBoxes.append(read_boxes_from_xml(xml))
    return AllBoxes


def read_img_boxes(boxStr, Img, allowed=['chair', 'person'], n=4):
    txt = boxStr.split('|')
    name = txt[0]
    boxes = txt[1:]
    if Img != None:
        if name != Img:
            return None, None, None
    l = []
    names = []
    for box in boxes:
        res = box.split(',')
        if allowed is not None and res[0] not in allowed:
            continue

        b = res[-n:]
        l.append(b)
        names.append(res[0])
    return l, names, name


def read_boxes(fileName, Img=None, n=4, allowed=['chair', 'person']):
    with open(fileName) as file:
        imgStrs = file.read().strip().split('\n')
        allBoxes = []
        allNames = []
        img_names = []
        for imgStr in imgStrs:
            res, names, img_name = read_img_boxes(imgStr, Img, n=n, allowed=allowed)
            if res != None:
                allBoxes.append(res)
                allNames.append(names)
                img_names.append(img_name)
        return allBoxes, allNames, np.array(img_names)


def read_boxes_from_txt(paths, delimeter=' ', allowed_objects=None):
    """

    Parameters
    ----------
    paths: List
        Paths of txt files, 1 for each each image
    delimeter: string
        Value separator of a box line
    allowed_objects: List or None
        Object that will be included in return results, if None all objects are included
    Returns
    -------
    3D List
        List of boxes which can be accessed as Boxes[img_ind][obj_no]
    2D List
        List of names which can be accessed as Names[img_ind][obj_no]
    """
    AllBoxes = []
    AllNames = []
    if allowed_objects is not None:
        allowed_objects = set(allowed_objects)  # Hashing to search in O(1) on average
    for path in paths:
        with open(path, 'r') as file:
            lines = file.read().strip().split('\n')
        obj_names = []
        boxes = []
        for line in lines:
            values = line.split(delimeter)
            obj_name = values[0]
            box = values[-4:]
            # box=[box[1],box[0],box[3],box[2]]
            if  (allowed_objects is not None) and (obj_name not in allowed_objects):
                continue
            boxes.append(box)
            obj_names.append(obj_name)
        AllBoxes.append(boxes)
        AllNames.append(obj_names)
    return AllBoxes, AllNames

def read_boxes_from_txt2(paths, delimeter=' ', allowed_objects=None):
    """

    Parameters
    ----------
    paths: List
        Paths of txt files, 1 for each each image
    delimeter: string
        Value separator of a box line
    allowed_objects: List or None
        Object that will be included in return results, if None all objects are included
    Returns
    -------
    3D List
        List of boxes which can be accessed as Boxes[img_ind][obj_no]
    2D List
        List of names which can be accessed as Names[img_ind][obj_no]
    """
    AllBoxes = []
    AllNames = []
    if allowed_objects is not None:
        allowed_objects = set(allowed_objects)  # Hashing to search in O(1) on average
    for path in paths:
        with open(path, 'r') as file:
            lines = file.read().strip().split('\n')
        obj_names = []
        boxes = []
        for line in lines:
            values = line.split(delimeter)
            obj_name = values[0]
            box = values[-4:]
            box=[box[1],box[0],box[3],box[2]]
            if  (allowed_objects is not None) and (obj_name not in allowed_objects):
                continue
            boxes.append(box)
            obj_names.append(obj_name)
        AllBoxes.append(boxes)
        AllNames.append(obj_names)
    return AllBoxes, AllNames

class VideoReader:
    def __init__(self, path, step_size=0, reshape_size=(512, 512)):
        self.path = path
        self.step_size = step_size
        self.curr_frame_no = 0
        self.video_finished = False
        self.reshape_size = reshape_size

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        return self

    def read(self):
        success, frame = self.cap.read()
        if not success:
            self.video_finished = True
            return success, frame
        for _ in range(self.step_size - 1):
            s, f = self.cap.read()
            if not s:
                self.video_finished = True
                break
        return success, frame

    def read_all(self):
        frames_list = []
        while not self.video_finished:
            success, frame = self.read()
            if success:
                # frame = resize(frame , self.reshape_size ) * 255).astype(np.uint8)
                frames_list.append(frame)

        return frames_list

    def __exit__(self, a, b, c):
        self.cap.release()
        cv2.destroyAllWindows()

