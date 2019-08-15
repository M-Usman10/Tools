import matplotlib.pyplot as plt
from PIL import Image


def display_landmarks(images, keypoints_labels):
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
