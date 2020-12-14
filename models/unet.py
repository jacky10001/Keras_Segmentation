from tensorflow.keras import Model, layers
from tensorflow.keras.applications import vgg16


def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = layers.Input(shape=(input_height, input_width, 3))

    base = vgg16.VGG16(include_top=False,
                       weights='imagenet',
                       input_tensor=img_input)
    
    base_out = base.output
    b4 = base.get_layer(name="block4_pool").output
    b3 = base.get_layer(name="block3_pool").output
    b2 = base.get_layer(name="block2_pool").output
    b1 = base.get_layer(name="block1_pool").output
    
    x = layers.UpSampling2D((2, 2))(base_out)
    x = layers.concatenate([b4, x], axis=-1)
    x = layers.Conv2D(512, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([b3, x], axis=-1)
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([b2, x], axis=-1)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([b1, x], axis=-1)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(nClasses, (1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Reshape((-1, nClasses))(x)
    x = layers.Activation("softmax")(x)
    
    return Model(inputs=img_input, outputs=x)


if __name__ == '__main__':
    m = UNet(15, 320, 320)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_unet.png')
    print(len(m.layers))
    m.summary()
