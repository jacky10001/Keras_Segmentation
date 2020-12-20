# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Tue Dec 15 01:53:25 2020
"""

import os
import tensorflow as tf

import load_batch
import models

ds = "dataset1"
input_height = 320
input_width = 320
n_classes = 11

key = "unet"
epochs = 500
lr = 0.001
train_batch_size = 10
val_batch_size = 10

train_images_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/images_prepped_train/"
train_segs_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/annotations_prepped_train/"

val_images_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/images_prepped_test"
val_segs_path = "D:/YJ/MyDatasets/Segmentation/data/"+ds+"/annotations_prepped_test"

method = {
    "fcn8": models.FCN8,
    "fcn32": models.FCN32,
    'segnet': models.SegNet,
    'unet': models.UNet,
}


#%%
G_tra = load_batch.imageSegmentationGenerator(
    train_images_path, train_segs_path, train_batch_size,
    n_classes=n_classes, input_height=input_height, input_width=input_width)

G_val = load_batch.imageSegmentationGenerator(
    val_images_path, val_segs_path, val_batch_size,
    n_classes=n_classes, input_height=input_height, input_width=input_width)


#%%
m = method[key](n_classes, input_height=input_height, input_width=input_width)
m.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    metrics=['acc']
)


#%%
os.makedirs('output_%s/tensorboard'%key, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="output_%s/model.h5"%key,
        monitor='acc', mode='auto', save_best_only=True,
        save_weights_only='True'),
    tf.keras.callbacks.TensorBoard(log_dir='output_%s/tensorboard'%key,
                                   histogram_freq=1,
                                   write_grads=True,
                                   write_images=True)
]

m.fit(G_tra,
      steps_per_epoch=367//train_batch_size,
      epochs=epochs,
      callbacks=callbacks,
      validation_data=G_val,
      validation_steps=101//val_batch_size
)
