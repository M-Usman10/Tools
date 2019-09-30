# from Tools.postprocessing.visualize import visualize_boxes_in_dir
#
# visualize_boxes_in_dir("/home/usman/Subset1")
# from preprocessing.converters import dense_box_to_file_box
# dense_box_to_file_box("/home/usman/PycharmProjects/keras-yolo3/Data/OpenImages/Subset1/Txt_results/boxes.txt","/home/usman/Desktop/experiment")

from preprocessing.converters import dense_box_to_file_box

# import numpy as np
# Inp_path="/home/usman/PycharmProjects/keras-yolo3/Data/OpenImages/Subset{}/Txt_results/boxes.txt"
# OutPath="/home/usman/PycharmProjects/keras-yolo3/Data/OpenImages/Subset{}/labels"
# all_names=[]
# found=np.load("Found.npy")
# for i in range(1,11):
#     dense_box_to_file_box(Inp_path.format(i),OutPath.format(i),allowed_objects=found)
from preprocessing.readers import read_images
from preprocessing.resizers import resize_image
def preprocess(img, shape=(416,416,3)):
    img=resize_image(img, shape[0], shape[1])
    return img[0],img[2],img[3]
imgs, imgs_names = read_images("/home/usman/PycharmProjects/keras-yolo3/Data/OpenImages/Subset1", format=".jpg", sorted=False, preprocess=preprocess, total=5)
print(imgs)