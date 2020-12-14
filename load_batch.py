# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Tue Dec 15 01:50:51 2020
"""

import os
import numpy as np
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import random


def getImageArr(im):

    img = im.astype(np.float32)

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img


def ungetImageArr(im):

    img = im.astype(np.float32)

    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    return img


def getSegmentationArr(seg, nClasses, input_height, input_width):

    seg_labels = np.zeros((input_height, input_width, nClasses))

    for c in range(nClasses):
        seg_labels[:, :, c] = (seg == c).astype(int)

    seg_labels = np.reshape(seg_labels, (-1, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size,
                               n_classes, input_height, input_width):
    
    images = sorted(
        glob.glob(os.path.join(images_path,"*.jpg")) +
        glob.glob(os.path.join(images_path,"*.png")) +
        glob.glob(os.path.join(images_path,"*.jpeg")))
    
    segmentations = sorted(
        glob.glob(os.path.join(segs_path,"*.jpg")) + 
        glob.glob(os.path.join(segs_path,"*.png")) +
        glob.glob(os.path.join(segs_path,"*.jpeg")))

    zipped = itertools.cycle(zip(images, segmentations))
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 0)

            assert im.shape[:2] == seg.shape[:2]

            assert im.shape[0] >= input_height and im.shape[1] >= input_width

            xx = random.randint(0, im.shape[0] - input_height)
            yy = random.randint(0, im.shape[1] - input_width)

            im = im[xx:xx + input_height, yy:yy + input_width]
            seg = seg[xx:xx + input_height, yy:yy + input_width]

            X.append(getImageArr(im))
            Y.append(getSegmentationArr(seg, n_classes, input_height, input_width))

        yield np.array(X), np.array(Y)


if __name__ == '__main__':
    G = imageSegmentationGenerator(
        "D:/YJ/MyDatasets/Segmentation/data/dataset1/images_prepped_train/",
        "D:/YJ/MyDatasets/Segmentation/data/dataset1/annotations_prepped_train/",
        batch_size=16, n_classes=15, input_height=320, input_width=320)
    x, y = G.__next__()
    
    x_show = ungetImageArr(x[0, ...]).astype('uint8')
    y_show = y[0, ..., 0].reshape(320,320)
    
    plt.imshow(x_show); plt.show()
    plt.imshow(y_show); plt.show()
    print(x.shape, y.shape)
