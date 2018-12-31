from random import shuffle
import glob
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

dirsep = '/'
csvdelim = ','
pathData='./ogle/data'
pathWeight = './models/cnn.h5'  # The HDF5 weight file generated for the trained model
pathModel = './models/cnn.nn'  # The model saved as a JSON file

DATAPATH='./ogle/data'
STEPS = 400
shuffle_data = True  # shuffle the addresses before saving
dataset_path = './[IV]/*.dat'
#manifest_path = './OGLE-CEP.txt'
manifest_path = './manifest.txt'
# read addresses and labels from the 'train' folder
addrs = glob.glob(dataset_path)
#labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
labels = np.zeros(len(addrs) - 1)
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    
##temp = float_data[:, 1] # <1> temperature (in degrees C)
#plt.plot(range(200000), temp[:200000])
###plt.show()
#
##plt.plot(range(1440), temp[:1440])
##plt.show()
#
#mean = float_data[:200000].mean(axis=0)
#float_data -= mean
#std = float_data[:200000].std(axis=0)
#float_data /= std
#norm = float_data[:, 1] # <1> temperature (in degrees C)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
valid_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
valid_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

def load_manifest(path):
    manifest = np.loadtxt(path, dtype={'names': ('star', 'field', 'starID', 'ra', 'dec', 'typ', 'subtyp', 'Imag', 'Vmag', 'pd', 'Iamp'), 'formats': ('S18', 'S10', 'i4', 'f4', 'f4', 'S8', 'S8', 'f4', 'f4', 'f4', 'f4')})
    return manifest

#path = './I/OGLE-LMC-CEP-1111.dat'
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

def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

manifest = load_manifest(manifest_path)

def get_period(manifest, path):
  stars = manifest['star']
  key = path.split('/')[2]
  key = key.split('.')[0]
  ind = np.searchsorted(manifest['star'],key)
  #print(key, ind, manifest['star'][ind], manifest['pd'][ind])
  # if ind is zero check whether it really is the first star in the list
  # return 0.0 on error
  #if ind == 0:
  #  if  key != manifest['star'][0]:
  #    return 0.0
  return manifest['pd'][ind], ind



train_filename = '%s/train.tfr'%(DATAPATH)  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    lightcurve = load_star(train_addrs[i])
    lightcurve = normalize(lightcurve)
    lightcurve = pad_or_truncate(lightcurve, STEPS)
    x = []
    list(x.extend(row) for row in lightcurve)
    y, star = get_period(manifest, train_addrs[i])
    #print("Star %s Period %f"%(star, y))
    # Create a feature
    feature = {'train/period': _float_feature(y),
               'train/data': _floatvector_feature(x)
              }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

# open the TFRecords file
valid_filename = '%s/valid.tfr'%(DATAPATH)  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(valid_filename)
for i in range(len(valid_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Val data: {}/{}'.format(i, len(valid_addrs)))
        sys.stdout.flush()
    # Load the image
    lightcurve = load_star(valid_addrs[i])
    lightcurve = normalize(lightcurve)
    lightcurve = pad_or_truncate(lightcurve, STEPS)
    x = []
    list(x.extend(row) for row in lightcurve)
    y, star = get_period(manifest, valid_addrs[i])
    #print("Star %s Period %f"%(star, y))
    # Create a feature
    feature = {'train/period': _float_feature(y),
               'train/data': _floatvector_feature(x)
              }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

# open the TFRecords file
test_filename = '%s/test.tfr'%(DATAPATH)  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
        sys.stdout.flush()
    # Load the image
    lightcurve = load_star(test_addrs[i])
    lightcurve = normalize(lightcurve)
    lightcurve = pad_or_truncate(lightcurve, STEPS)
    x = []
    list(x.extend(row) for row in lightcurve)
    y, star = get_period(manifest, test_addrs[i])
    #print("Star %s Period %f"%(star, y))
    # Create a feature
    feature = {'train/period': _float_feature(y),
               'train/data': _floatvector_feature(x)
              }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()


