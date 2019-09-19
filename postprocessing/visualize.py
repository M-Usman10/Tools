import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from ..postprocessing.detection import non_max_suppression
from ..preprocessing.readers import read_boxes, read_images
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


def visualize(img,boxes,names=None,dict_=None,vis=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,box in enumerate(boxes):
        top, left, bottom, right=box.astype(float).astype(int)
        if dict_ is not None and names[i] in dict_.keys():
            cv2.putText(img, names[i], (left, top), font, .4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img,(left,top),(right,bottom),dict_[names[i]], 1)
        else:
            if names is not None:
                cv2.putText(img, names[i], (left, top), font, .4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (left, top), (right, bottom), (255,255,255), 1)
    if vis:
        plt.subplots(figsize=(6, 6))
        plt.imshow(img[...,::-1])
        plt.show()
    return img


def visualize_boxes_in_dir(imgs_path, label_path, allowed_objs=None, img_format='.jpg', img_shape=(416, 416, 3),
                           nms=0.8):
    """Directory containing images and one txt file for boxes"""
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

    # for i in range(len(imgs)):
    #     pick = non_max_suppression(np.array(boxes[i]).astype(float).astype(int), 0.8)
    #     res.append(visualize(imgs[i].copy(), np.array(boxes[i])[pick], names=np.array(names[i])[pick], dict_=colors))
    return res
