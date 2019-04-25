import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling2D, concatenate, ELU, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from Loader import Loader
import config

def build_model(with_compile = True):
    # Building the CNN
    regularizer = l2(1e-5)

    input_shape = tuple(config.IMG_DIMS)

    inputs_ph = keras.Input(shape=input_shape)
    x = inputs_ph

    # Layer 1
    x = Conv2D(16, (5,5), padding='same', kernel_regularizer=regularizer,
            name = 'conv_1')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='MP_1')(x)

    # Layer 2
    x = Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizer,
            name = 'conv_2')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_2')(x)
    x = Dropout(0.1)(x)

    # Layer 3
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizer,
            name = 'conv_3')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_3')(x)

    # Layer 4
    x = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizer,
            name = 'conv_4')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_4')(x)
    x = Dropout(0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizer,
            name = 'conv_5')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_5')(x)

    # Layer 6
    x = Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizer,
            name = 'conv_6')(x)
    x = ReLU()(x)

    # layer 7
    x = Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizer,
            name = 'conv_7')(x)
    x = BatchNormalization(axis=3)(x)
    x = ReLU()(x)

    # # GAP
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    # Dense
    x = Dense(256, kernel_regularizer=regularizer, name = 'dense')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    # output
    x = Dense(1, kernel_regularizer=regularizer, name='output')(x)

    # model
    model = Model(inputs_ph, x)


    if with_compile:
        optimizer = Adam(lr = 0.01)
        model.compile(optimizer=optimizer,
                        loss = 'mean_squared_error',
                        metrics = ['mean_absolute_error'])
    return model


if __name__ == '__main__':
    loader_batch_size = 8
    num_epochs = 5

    loader = Loader([config.LABELS_ONLY], loader_batch_size)
    model = build_model()


    for epoch in range(num_epochs):
        batch = loader.next_batch()
        while batch:
            imgs = batch[0]
            years = batch[1]

            loss = model.train_on_batch(imgs, years)
            print(loss)
            batch = loader.next_batch()