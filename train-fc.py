import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

ZDim=3
XDim=10
YDim=40

sizeBatch=128
sizeValid=2000
nEpochs=500
nExamples=9589

pathTrain = './ogle/data/train.tfr'  # The TFRecord file containing the training set
pathValid = './ogle/data/valid.tfr'    # The TFRecord file containing the validation set
pathTest = './ogle/data/test.tfr'    # The TFRecord file containing the test set
pathWeight = './models/cnn.h5'  # The HDF5 weight file generated for the trained model
pathModel = './models/cnn.nn'  # The model saved as a JSON file

def PNet():
  inputs = Input((YDim, XDim, ZDim))
  #conv1 = Conv3D(32, (WDim, 5, 5), activation='relu', padding='same')(inputs)
  #conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1)
  #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  #inputs = Reshape((40,10,3))(inputs)
  conv1 = Conv2D(512, (10, 10), activation='relu', padding='same')(inputs)
  f1 = Flatten()(conv1)
  fc1 = Dense(1024, activation='linear')(f1)
  fc2 = Dense(2048, activation='linear')(fc1)
  fc3 = Dense(4192, activation='linear')(fc2)
  fc4 = Dense(8096, activation='linear')(fc3)
  fc5 = Dense(4192, activation='linear')(fc4)
  period = Dense(1, activation='linear')(fc5)

  model = Model(inputs=[inputs], outputs=[period])

  model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse', 'mae'])

  print(model.summary())

  return model


def train():
  K.set_image_data_format('channels_last')  # TF dimension ordering in this code
  featdef = {
    'train/data': tf.FixedLenFeature(shape=[1200], dtype=tf.float32),
    'train/period': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    }
        
  def _parse_record(example_proto, clip=False):
    """Parse a single record into x and y images"""
    example = tf.parse_single_example(example_proto, featdef)
    x = example['train/data']
    x = tf.reshape(x, (YDim, XDim, ZDim))
    y = example['train/period']

    return x, y

  #construct a TFRecordDataset
  dsTrain = tf.data.TFRecordDataset(pathTrain).map(_parse_record)
  dsTrain = dsTrain.shuffle(1000)
  dsTrain = dsTrain.repeat()
  dsTrain = dsTrain.batch(sizeBatch)

  dsValid = tf.data.TFRecordDataset(pathValid).map(_parse_record)
  dsValid = dsValid.shuffle(1000)
  dsValid = dsValid.repeat()
  dsValid = dsValid.batch(sizeValid)

  #dsTest = tf.data.TFRecordDataset(pathTest).map(_parse_record)
  #dsTest = dsValid.repeat(30)
  #dsTest = dsValid.shuffle(10).batch(sizeBatch)

  print('-'*30)
  print('Creating and compiling model...')
  print('-'*30)
  model = PNet()
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(pathWeight, verbose=1, save_best_only=True),
      tf.keras.callbacks.TensorBoard(log_dir='./logs')
  ]

  print('-'*30)
  print('Fitting model...')
  print('-'*30)

  #print(dsTrain)
  history = model.fit(dsTrain, validation_data=dsValid, validation_steps=1, steps_per_epoch=int(np.ceil(nExamples/sizeBatch)), epochs=nEpochs, verbose=1, callbacks=callbacks)
  #history = model.fit(dsTrain, validation_data=dsValid, epochs=nEpochs, verbose=1, callbacks=callbacks)

  # serialize model to JSON
  model_serial = model.to_json()
  with open(pathModel, "w") as yaml_file:
    yaml_file.write(model_serial)


print(tf.__version__)
train()
