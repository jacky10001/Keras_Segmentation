# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Tue Dec 15 01:53:58 2020
"""

import os
import glob
import numpy as np
import cv2
import random
from skimage import color, exposure


def imageSegmentationGenerator(images_path, segs_path, n_classes):

    images = sorted(
        glob.glob(os.path.join(images_path,"*.jpg")) +
        glob.glob(os.path.join(images_path,"*.png")) +
        glob.glob(os.path.join(images_path,"*.jpeg")))
    
    segmentations = sorted(
        glob.glob(os.path.join(segs_path,"*.jpg")) + 
        glob.glob(os.path.join(segs_path,"*.png")) +
        glob.glob(os.path.join(segs_path,"*.jpeg")))
    
    assert len(images) == len(segmentations)
    
    
    colors = []
    for _ in range(n_classes):
        colors.append((random.randint(0, 255),
                       random.randint(0, 255),
                       random.randint(0, 255)))
                      
    for im_fn, seg_fn in zip(images, segmentations):
        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print(np.unique(seg))

        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                                 (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                                 (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                                 (colors[c][2])).astype('uint8')

        eqaimg = color.rgb2hsv(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        eqaimg[:, :, 2] = exposure.equalize_hist(eqaimg[:, :, 2])
        eqaimg = color.hsv2rgb(eqaimg)

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cv2.imshow("equalize_hist_img",
                   cv2.cvtColor(
                       (eqaimg * 255.).astype(np.uint8),
                       cv2.COLOR_RGB2BGR))
        
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imageSegmentationGenerator(
        images_path="D:/YJ/MyDatasets/Segmentation/data/dataset2/images_prepped_train",
        segs_path="D:/YJ/MyDatasets/Segmentation/data/dataset2/annotations_prepped_train",
        n_classes = 11
)


