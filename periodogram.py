import os
import numpy as np
import matplotlib.pyplot as plt
import fs
import glob
import sys
import tensorflow as tf

from fs import open_fs
from random import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

dirsep = '/'
csvdelim = ','
#pathData='./ogle/data'
pathData='./[IV]/*.dat'
pathWeight = './models/cnn.h5'  # The HDF5 weight file generated for the trained model
pathModel = './models/cnn.nn'  # The model saved as a JSON file
pathManifest = "./manifest.txt" # Master list of all stars and their meta properties
STEPS = 400

def load_manifest(path):
    manifest = np.loadtxt(path, dtype={'names': ('star', 'field', 'starID', 'ra', 'dec', 'typ', 'subtyp', 'Imag', 'Vmag', 'pd', 'Iamp'), 'formats': ('S18', 'S10', 'i4', 'f4', 'f4', 'S8', 'S8', 'f4', 'f4', 'f4', 'f4')})
    return manifest

def load_star(path):
    lightcurve = np.loadtxt(path, dtype={'names': ('jd', 'mag', 'err'), 'formats': ('f4', 'f4', 'f4')})
    return lightcurve

def normalize(x):
    x['jd'] = x['jd'] - x['jd'][0]
    min_mag = np.min(x['mag'])
    x['mag'] = x['mag'] - min_mag
    return x

def pad_or_truncate(x, steps):
    z = (0.0, 0.0, 0.0)
    n=x.shape[0]
    if n > steps:
        x = x[:steps]
    if n < steps:
        x = np.append(x, x[0])
        x[n] = z
        n += 1
        if n < steps:
            z = np.tile(x[n-1],steps-n)
            x = np.append(x, z)
    return x

def get_period(manifest, path):
  stars = manifest['star']
  key = path.split('/')[2]
  key = key.split('.')[0]
  ind = np.searchsorted(manifest['star'],key)
  return manifest['pd'][ind], ind


def processLC(pathManifest, pathData):
  shuffle_data = False  # shuffle the addresses before predicting
  # read addresses and labels from the data path
  addrs = glob.glob(pathData)
  #labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
  labels = np.zeros(len(addrs) - 1)
  # to shuffle data
  if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
  manifest = load_manifest('./manifest.txt')
  for i in range(len(addrs)):
    # Load the image
    lightcurve = load_star(addrs[i])
    lightcurve = normalize(lightcurve)
    lightcurve = pad_or_truncate(lightcurve, STEPS)
    x = []
    list(x.extend(row) for row in lightcurve)
    y, ind = get_period(manifest, addrs[i])
    yield x, y, manifest['star'][ind]

#------------------------

# main
# load the trained model
# load from YAML and create model
model_file = open(pathModel, 'r')
model_serial = model_file.read()
model_file.close()
model = model_from_json(model_serial)
# load weights into new model
model.load_weights(pathWeight)
print("Loaded model from disk")
model.summary()
 
# evaluate loaded model on test data
model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse','mae'])

# global plot settings
Nr = 1
Nc = 1

# find input files in the target dir "pathData"
for x, y, star in processLC("./manifest.txt", pathData):
  plt.gray()
  # forward pass through the model to invert the Level 1 SP images and predict the Level 2 image mag fld maps
  batchIn = np.reshape(x[:], (1, 400, 1, 3))
  batchOut = model.predict(batchIn)
  period = batchOut[0,0]

  # prepare to visualize
  fig1 = plt.figure(1)
  fig1.suptitle('%s P=%f'%(star, y), fontsize=14)
  #plt.subplot(11)
  plt.title('P pred: %.2f'%(period))
  lc = np.reshape(batchIn,(-1, 400, 3))
  lc[0,:,0] = lc[0,:,0] % period
  Range = lc[0,:,1]
  Series = lc[0,:,0]
  print("Star %s P=%f, P^=%f"%(star, y, period))
  plt.plot(Series, Range, 'bo', markersize=2)
  plt.show()
  plt.close(fig1)
