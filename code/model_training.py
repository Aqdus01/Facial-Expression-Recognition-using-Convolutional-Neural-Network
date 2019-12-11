# %% --------------------------------------- Imports -------------------------------------------------------------------
import pandas as pd
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator

# %% ---------------------------------------- Set-Up -------------------------------------------------------------------
SEED = 666
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

image_size = (54, 72)

# Read training data in for Image Generator ----------------------------------------------------------------------------
train_df = pd.read_csv('data/training_images_preprocessed.csv')

image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=0.25, horizontal_flip=True)

train_generator = image_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="training",
    batch_size=256,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)

valid_generator = image_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=image_size)

# Create Convolutional Neural Net Architecture -------------------------------------------------------------------------
# All parameters were defined from tuning
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_size[0], image_size[1], 3), padding='same'))
model.add(Activation('elu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('tanh'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(Adam(lr=0.001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

# checkpoints to avoid overfitting
checkpoint = ModelCheckpoint('model_wang.hdf5', monitor='val_accuracy', save_best_only=True, mode='max',
                             verbose=1)

callbacks_list = [checkpoint]

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

# Fit the compiled model
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=callbacks_list,
                    epochs=30)
