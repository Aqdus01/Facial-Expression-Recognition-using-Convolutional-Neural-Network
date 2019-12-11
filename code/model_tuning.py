# %% --------------------------------------- Imports -------------------------------------------------------------------
import pandas as pd
import os
import talos
import random
import numpy as np
import tensorflow as tf
from platform import python_version_tuple
from keras.initializers import glorot_uniform
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from talos.model.normalizers import lr_normalizer
from talos.model.early_stopper import early_stopper
from talos import Evaluate

# Resource used: https://github.com/autonomio/talos

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
test_df = pd.read_csv('data/validation_images_preprocessed.csv')

image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=0.25, horizontal_flip=True)

train_generator = image_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="name",
    y_col="class",
    subset="training",
    batch_size=32,
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

test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="name",
    y_col="class",
    target_size=image_size,
    batch_size=32,
    seed=SEED,
    class_mode='categorical')

#  ----------------------------------- Hyper Parameters --------------------------------------------------------------
p = {'lr': (0.0001, 10, 10),
     'neurons_layer_2': [32, 64, 128],
     'neurons_layer_3': [32, 64, 128, 256],
     'neurons_layer_4': [32, 64, 128, 256],
     'batch_size': [256, 512],
     'epochs': [30],
     'dropout': (0, 0.50, 10),
     'kernel_initializer': ['uniform', 'normal', 'random_uniform'],
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': [Adam, Nadam, RMSprop, SGD],
     'loss': ['categorical_crossentropy'],
     'activation_1': ['relu', 'elu', 'tanh'],
     'activation_2': ['relu', 'elu', 'tanh'],
     'activation_3': ['relu', 'elu', 'tanh'],
     'activation_4': ['relu', 'elu', 'tanh'],
     'activation_5': ['relu', 'elu', 'tanh'],
     'last_activation': ['softmax']}


#  -------------------------------------- CNN Tuning ----------------------------------------------------------

def expression_model(trainX, trainY, testX, testY, params):
    model = Sequential()
    # Input Layer
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer=params['kernel_initializer']))
    model.add(Activation(params['activation_1']))
    # Layer 2 (Hidden)
    model.add(Conv2D(params['neurons_layer_2'], (3, 3)))
    model.add(Activation(params['activation_2']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    # Layer 3 (Hidden)
    model.add(Conv2D(params['neurons_layer_3'], (3, 3), padding='same'))
    model.add(Activation(params['activation_3']))
    # Layer 4 (Hidden)
    model.add(Conv2D(params['neurons_layer_4'], (3, 3)))
    model.add(Activation(params['activation_4']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout']))
    # Layer 5 (Hidden)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(params['activation_5']))
    model.add(Dropout(params['dropout']))
    # Output Layer (7 classifications)
    model.add(Dense(7, activation=params['last_activation']))

    # Tune Optimization/Learning Rate
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'], metrics=['accuracy'])

    # Fit model
    history = model.fit(trainX, trainY, batch_size=params['batch_size'], epochs=params['epochs'],
                        validation_data=(testX, testY),
                        callbacks=[ModelCheckpoint("conv2d_wang.hdf5", monitor="val_loss", save_best_only=True),
                                   early_stopper(params['epochs'], mode='strict')])

    return history, model


#  ------------------------------------ Get Data from Image Generator---------------------------------------------------

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

tempX, tempY = izip(*(train_generator[i] for i in xrange(len(train_generator))))
trainX, trainY = np.vstack(tempX), np.vstack(tempY)
del tempX, tempY

tempX, tempY = izip(*(valid_generator[i] for i in xrange(len(valid_generator))))
testX, testY = np.vstack(tempX), np.vstack(tempY)
del tempX, tempY
#  ------------------------------------------ CNN Tuning Eval --------------------------------------------------------

tuning = talos.Scan(x=trainX,
                    y=trainY,
                    x_val=testX,
                    y_val=testY,
                    model=expression_model,
                    params=p,
                    experiment_name='emotional_classification',
                    round_limit=100)  # just does 100 rounds of modeling / 100 different param configs
# fraction_limit=.10)  # just does 10% of total number param configs)

# Convert tuning results to Pandas data-frame and save it
results = Evaluate(tuning)
results_df = results.data
results_df = results_df.sort_values(by='val_accuracy', ascending=True)
results_df.to_csv(r'/home/ubuntu/Project/tuning_results.csv')
