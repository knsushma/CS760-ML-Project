import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling2D, concatenate, ELU, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from Loader import Loader
import config
import sys

def build_model(learning_rate, with_compile = True, model_type = 'multitask'):
    # Building the CNN
    regularizer = l2(1e-5)
#     regularizer = l2(0)

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
    # Year
    if model_type == 'year':
        x = Dense(1, kernel_regularizer = regularizer, name = 'output')(x)
        model = Model(inputs_ph, x)
   
    # Genre

    if model_type == 'genre':
        x = Dense(8, kernel_regularizer=regularizer, name='output')(x)
        x = Activation('softmax')(x)
        model = Model(inputs_ph, x)

    # Artist
    if model_type == 'artist':
        x = Dense(config.NUM_ARTISTS, kernel_regularizer = regularizer, name = 'output')(x)
        x = Activation('softmax')(x)
        model = Model(inputs_ph, x)


    # model
    if model_type == 'multitask':
        year = Dense(1, kernel_regularizer = regularizer, name = 'year')(x)

        genre = Dense(8, kernel_regularizer=regularizer, name='genre_activation')(x)
        genre = Activation('softmax', name = 'genre')(genre)
        artist = Dense(config.NUM_ARTISTS, kernel_regularizer = regularizer, name = 'artist_activation')(x)
        artist = Activation('softmax', name = 'artist')(artist)  
        model = Model(inputs_ph, outputs = [year, genre, artist])

    if with_compile:
        optimizer = Adam(lr = learning_rate)

        if model_type == 'year':
                model.compile(optimizer=optimizer,
                        loss = 'mean_squared_error',
                        metrics = ['mean_absolute_error'])
        if model_type == 'genre' or model_type == 'artist':
                model.compile(optimizer=optimizer,
                        loss = 'categorical_crossentropy',
                        metrics=['accuracy'])
        if model_type == 'multitask':
                model.compile(optimizer=optimizer,
                        loss={'genre': 'categorical_crossentropy', 'artist': 'categorical_crossentropy',
                                'year': 'mean_squared_error'},
                        metrics={'genre': 'accuracy', 'artist': 'accuracy', 'year': 'mean_absolute_error'})

    return model


if __name__ == '__main__':
    '''
        Args:
        argv[1] - Either 'validate' to use a train/validation fold, or 'test' to use train/test set
        argv[2] - Learning Rate (float)
        argv[3] - either 'year', 'genre', 'artist', or 'multitask' for the task type
        argv[4] - The number of epochs to train on
    '''
    loader_batch_size = 8

    if sys.argv[1] == 'validate':
        train_folds = [config.FULL_TRAIN_LABELS]
        validation_folds = [config.FULL_TEST_LABELS]
    else:
        train_folds = [config.TRAIN_FOLDS_LABELS[0]]
        validation_folds = [config.TEST_FOLDS_LABELS[0]]

    learning_rate = float(sys.argv[2])
    model_type = sys.argv[3]
    num_epochs = int(sys.argv[4])

    loader_train = Loader(train_folds, loader_batch_size)
    loader_validation = Loader(validation_folds, loader_batch_size)
    model = build_model(learning_rate)

    for epoch in range(num_epochs):
        batch = loader_train.next_batch()

        if model_type == 'multitask':
            losses = [0] * 7
        else:
            losses = [0, 0]
        
        batch_N = 0
        while batch:
            batch_N += 1
            imgs = batch[0]
            years = batch[1]
            genre = batch[2]
            artist = batch[3]

            if model_type == 'genre':
                loss = model.train_on_batch(imgs, genre)
            if model_type == 'year':
                loss = model.train_on_batch(imgs, years)
            if model_type == 'artist':
                loss = model.train_on_batch(imgs, artist)
            if model_type == 'multitask':
                loss = model.train_on_batch(imgs, {'year': years, 'genre': genre, 'artist': artist})
        
            if not model_type == 'multitask':
                losses[0] += loss[0]
                losses[1] += loss[1]
            else:
                for i in range(7):
                    losses[i] += loss[i]


            batch = loader_train.next_batch()
        
        print('Epoch: ', epoch + 1)
        if model_type == 'multitask':
            losses = [loss / batch_N for loss in losses]
            print('Losses:', *losses)
        else:
            print('Avg Acc: ', losses[0] / batch_N, losses[1] / batch_N)


    # Final Train Set Evaluation
    if model_type == 'multitask':
        losses = [0] * 7
    else:
        losses = [0,0]
    batch_N = 0
    batch = loader_train.next_batch()
    
    while batch:
        batch_N += 1
        imgs = batch[0]
        years = batch[1]
        genre = batch[2]
        artist = batch[3]
        
        if model_type == 'genre':
            loss = model.test_on_batch(imgs, genre)
        if model_type == 'year':
            loss = model.test_on_batch(imgs, years)
        if model_type == 'artist':
            loss = model.test_on_batch(imgs, artist)
        if model_type == 'multitask':
            loss = model.test_on_batch(imgs, {'year': years, 'genre': genre, 'artist': artist})

        if not model_type == 'multitask':
            losses[0] += loss[0]
            losses[1] += loss[1]
        else:
            for i in range(7):
                losses[i] += loss[i]

        batch = loader_train.next_batch()

    if model_type == 'multitask':
        losses = [loss / batch_N for loss in losses]
        print('Final Train Accuracy:', *losses)
    else:
        print('Final Train Accuracy: ', losses[0] / batch_N, losses[1] / batch_N)


    # Final Validation Set Accuracy
    if model_type == 'multitask':
        losses = [0] * 7
    else:
        losses = [0,0]
    batch_N = 0
    batch = loader_validation.next_batch()

    while batch:
        batch_N += 1
        imgs = batch[0]
        years = batch[1]
        genre = batch[2]
        artist = batch[3]

        if model_type == 'genre':
            loss = model.test_on_batch(imgs, genre)
        if model_type == 'year':
            loss = model.test_on_batch(imgs, years)
        if model_type == 'artist':
            loss = model.test_on_batch(imgs, artist)
        if model_type == 'multitask':
            loss = model.test_on_batch(imgs, {'year': years, 'genre': genre, 'artist': artist})

        if not model_type == 'multitask':
            losses[0] += loss[0]
            losses[1] += loss[1]
        else:
            for i in range(7):
                losses[i] += loss[i]
        batch = loader_validation.next_batch()

    if model_type == 'multitask':
        losses = [loss / batch_N for loss in losses]
        print('Final Validation Accuracy:', *losses)
    else:
        print('Final Validation Accuracy: ', losses[0] / batch_N, losses[1] / batch_N)
