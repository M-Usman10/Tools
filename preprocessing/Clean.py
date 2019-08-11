import numpy as np


def remove_outliers(data_list, idx, lim=3, attr_thresh=20):
    data = data_list[idx]
    mean = np.mean(data, axis=0)
    std = np.mean(data, axis=0)
    to_keep = np.where(~(np.sum((data <= (mean - lim * std)) | (data >= (mean + lim * std)), axis=1) > attr_thresh))[0]
    res = []
    for elem in data_list:
        res.append(elem[to_keep])
    return res
