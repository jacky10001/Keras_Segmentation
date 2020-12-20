# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Tue Dec 15 01:51:19 2020
"""

import os
import cv2
import glob
import random
import numpy as np

import load_batch
import models

ds = 'dataset1'
input_height = 320
input_width = 320
n_classes = 11

key = "unet"

images_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/images_prepped_test"
segs_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/annotations_prepped_test/"

method = {
    "fcn8": models.FCN8,
    "fcn32": models.FCN32,
    'segnet': models.SegNet,
    'unet': models.UNet,
}

images = sorted(
    glob.glob(os.path.join(images_path,"*.jpg")) +
    glob.glob(os.path.join(images_path,"*.png")) +
    glob.glob(os.path.join(images_path,"*.jpeg"))
)
    
segmentations = sorted(
    glob.glob(os.path.join(segs_path,"*.jpg")) + 
    glob.glob(os.path.join(segs_path,"*.png")) +
    glob.glob(os.path.join(segs_path,"*.jpeg"))
)


colors = []
for _ in range(n_classes):
    colors.append((random.randint(0, 255),
                   random.randint(0, 255),
                   random.randint(0, 255)))


def label2color(colors, n_classes, seg):
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((seg == c) *
                               (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((seg == c) *
                               (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((seg == c) *
                               (colors[c][2])).astype('uint8')
    seg_color = seg_color.astype(np.uint8)
    return seg_color


def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy
    


# m = load_model("output/%s_model.h5" % key)
m = method[key](11, input_height=input_height, input_width=input_width)
m.load_weights("output_%s/model.h5" % key)

for i, (imgName, segName) in enumerate(zip(images, segmentations)):
    print("%04d/%04d %s"%(i + 1, len(images), imgName))

    im = cv2.imread(imgName, 1)
    # im=cv2.resize(im,(input_height,input_width))
    xx, yy = getcenteroffset(im.shape, input_height, input_width)
    im = im[xx:xx + input_height, yy:yy + input_width, :]

    seg = cv2.imread(segName, 0)
    # seg= cv2.resize(seg,interpolation=cv2.INTER_NEAREST)
    seg = seg[xx:xx + input_height, yy:yy + input_width]

    pr = m.predict(np.expand_dims(load_batch.getImageArr(im), 0))[0]
    pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)

    cv2.imshow("img", im)
    cv2.imshow("seg_predict_res", label2color(colors, n_classes, pr))
    cv2.imshow("seg", label2color(colors, n_classes, seg))

    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()
