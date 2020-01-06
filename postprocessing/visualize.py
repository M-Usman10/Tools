import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from ..postprocessing.detection import non_max_suppression
from ..preprocessing.readers import read_boxes_from_txt, read_boxes,read_images
from ..preprocessing.resizers import resize_image

colors={"person":(255,255,0),"chair":(255,255,255),"bottle":(0,0,255),"Furniture":(0,0,255),"diningtable":(255,0,0),"Table":(255,0,0),"wineglass":(50,50,50),"cup":(0,0,0)}

def visualize_landmarks(images, keypoints_labels):
    WIDTH = 14
    HEIGHT = 10
    rows = len(images)
    fig = plt.figure(figsize=(WIDTH, HEIGHT * rows))
    columns = 2
    for i in range(len(images)):
        image = images[i]
        flattened_keypoints_label = keypoints_labels[i]
        keypoints_label = flattened_keypoints_label.reshape(int(flattened_keypoints_label.shape[0] / 3), 3)
        img = Image.fromarray(image[..., ::-1])
        ax = fig.add_subplot(rows, columns, i * 2 + 1, projection='3d')
        surf = ax.scatter(keypoints_label[:, 0] * 1.2, keypoints_label[:, 1], keypoints_label[:, 2], c="cyan",
                          alpha=1.0, edgecolor='b')
        ax.plot3D(keypoints_label[:17, 0] * 1.2, keypoints_label[:17, 1], keypoints_label[:17, 2], color='blue')
        ax.plot3D(keypoints_label[17:22, 0] * 1.2, keypoints_label[17:22, 1], keypoints_label[17:22, 2], color='blue')
        ax.plot3D(keypoints_label[22:27, 0] * 1.2, keypoints_label[22:27, 1], keypoints_label[22:27, 2], color='blue')
        ax.plot3D(keypoints_label[27:31, 0] * 1.2, keypoints_label[27:31, 1], keypoints_label[27:31, 2], color='blue')
        ax.plot3D(keypoints_label[31:36, 0] * 1.2, keypoints_label[31:36, 1], keypoints_label[31:36, 2], color='blue')
        ax.plot3D(keypoints_label[36:42, 0] * 1.2, keypoints_label[36:42, 1], keypoints_label[36:42, 2], color='blue')
        ax.plot3D(keypoints_label[42:48, 0] * 1.2, keypoints_label[42:48, 1], keypoints_label[42:48, 2], color='blue')
        ax.plot3D(keypoints_label[48:, 0] * 1.2, keypoints_label[48:, 1], keypoints_label[48:, 2], color='blue')
        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax = fig.add_subplot(rows, columns, i * 2 + 2)
        ax.imshow(img)
    plt.show()

def show_box(img,box):
  x1, y1, x2, y2 =box.astype(int)
  img=cv2.rectangle(img.copy(),(x1,y1),(x2,y2),(255,255,255),4)
  plt.imshow(img[...,::-1])
  plt.show()


def visualize(img,boxes,names=None,dict_=None,vis=True, write_names=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,box in enumerate(boxes):
        top, left, bottom, right=box.astype(float).astype(int)
        if dict_ is not None and names[i] in dict_.keys():
            if write_names:
                cv2.putText(img, names[i], (left, top), font, 2, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img,(left,top),(right,bottom),dict_[names[i]], 3)
        else:
            if write_names:
                cv2.putText(img, names[i], (left, top), font, 2, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (left, top), (right, bottom), (255,255,255), 3)
    if vis:
        plt.subplots(figsize=(6, 6))
        plt.imshow(img[...,::-1])
        plt.show()
    return img


def visualize_boxes_dense(imgs_path, label_path, allowed_objs=None, img_format='.jpg', img_shape=(416, 416, 3), nms=0.8):
    """

    Parameters
    ----------
    imgs_path: string
        Directory path containing Images
    label_path: string
        File containing boxes in Dense Format
    allowed_objs: List or None
        Objects types that will be loaded, if None all types will be loaded
    img_format: string
        Allowed Formats to read from Img directory e.g. ".jpg"
    img_shape: Tuple of 3 values
        Shape of Images used while box detection
    nms
        Non-maxim
    Returns
    -------
    List
        All the images with boxes drawn on them
    """
    import glob
    import numpy as np
    labels = glob.glob(os.path.join(label_path, "*.txt"))
    labels=labels[0]
    boxes, names_boxes, img_names = read_boxes(labels, allowed=allowed_objs)
    def preprocess(img,shape=img_shape):
        return resize_image(img,shape[0],shape[1])[0]
    imgs, names_imgs = read_images(imgs_path, format=img_format, sorted=False, preprocess=preprocess, total=5)
    res=[]
    for img_idx, name in enumerate(names_imgs):
        idx_box = np.where(name == img_names)[0][0]
        pick = non_max_suppression(np.array(boxes[idx_box]).astype(float).astype(int), 0.65)

        res.append(
            visualize(imgs[img_idx].copy(), np.array(boxes[idx_box])[pick], names=np.array(names_boxes[idx_box])[pick],
                      dict_=colors))
    return res


def visualize_boxes_txt(img_dir, boxes_dir, allowed_objs=["Table","Desk","Bottle"], img_format='.jpg', img_shape=(416, 416, 3), nms=0.8):
    """

        Parameters
        ----------
        img_dir: string
            Directory path containing Images
        boxes_dir: string
            File containing boxes in Dense Format
        allowed_objs: List or None
            Objects types that will be loaded, if None all types will be loaded
        img_format: string
            Allowed Formats to read from Img directory e.g. ".jpg"
        img_shape: Tuple of 3 values
            Shape of Images used while box detection
        nms
            Non-maxim
        Returns
        -------
        List
            All the images with boxes drawn on them
        """
    import numpy as np
    def preprocess(img, shape=img_shape):
        return img
    #     return resize_image(img, shape[0], shape[1])[0]
    imgs, imgs_names = read_images(img_dir, format=img_format, sorted=False, preprocess=preprocess, total=5)
    boxes_paths=[]
    for i in range(len(imgs_names)):
        boxes_paths.append(os.path.join(boxes_dir,imgs_names[i].split('.')[0]+'.txt'))
    boxes, obj_names = read_boxes_from_txt(boxes_paths, allowed_objects=allowed_objs)
    res = []
    for i in range(len(imgs_names)):
        pick = non_max_suppression(np.array(boxes[i]).astype(float).astype(int), nms)
        res.append(
            visualize(imgs[i].copy(), np.array(boxes[i])[pick], names=np.array(obj_names[i])[pick],
                      dict_=colors))
    return res