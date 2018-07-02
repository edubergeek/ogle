import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras._impl.keras.models import Sequential
#from keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers import Flatten, Dense, LSTM
#from keras.layers import Flatten, Dense, LSTM
from tensorflow.python.keras._impl.keras.optimizers import RMSprop
#from keras.optimizers import RMSprop
from tensorflow.python.keras._impl.keras import backend as K
#from keras import backend as K


model_name = 'test'
n_epochs = 10
n_steps = 10000
batch_size = 32
dropout = 0.0
rdropout = 0.0

train_path = './ogle/data/train.tfr'
valid_path = './ogle/data/valid.tfr'
test_path = './ogle/data/test.tfr'

def the_input_iterator(filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(serialized):
        features = {
            'train/data': tf.FixedLenFeature([1200], tf.float32),
	    'train/period': tf.FixedLenFeature([], tf.float32)
        }
	# Parse one training example
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
	# Get the period as y
        y = tf.cast(parsed_example['train/period'], tf.float32)

	# Get the light curve as x
        x = parsed_example['train/data']
        #x = tf.decode_raw(x_raw, tf.float32)
        #x = tf.cast(x, tf.float32)
        # reshape from 1200 floats to 400 rows of 3 values each
        x_shape = tf.stack([400, 3])
        x = tf.reshape(x, x_shape)
	    
        d={'lstm1_input':x},y
        return d
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def neuralnetwork():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, dropout=dropout, recurrent_dropout=rdropout, input_shape=(400, 3), name='lstm1'))
    model.add(LSTM(64, dropout=dropout, recurrent_dropout=rdropout))
    model.add(Dense(128, name='fc1'))
    model.add(Dense(1, name='prediction'))
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    model.summary()
    return model

def train():
    sess = tf.Session()
    K.set_session(sess)

    model = neuralnetwork()
    model_dir = os.path.join(os.getcwd(), "./models/%s"%(model_name))
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir: ",model_dir)
    the_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: the_input_iterator(train_path,
             perform_shuffle=True, repeat_count=n_epochs, batch_size=batch_size), max_steps=n_steps)
    valid_spec = tf.estimator.EvalSpec(input_fn=lambda: the_input_iterator(valid_path, perform_shuffle=False, batch_size=1))

    tf.estimator.train_and_evaluate(the_estimator, train_spec, valid_spec)

def test(Estimator):
    predict_results = Estimator.predict( input_fn=lambda: the_input_iterator(test_path, perform_shuffle=False, batch_size=1))
    y_classes = keras.np_utils.probas_to_classes(y_proba)
    predicted = []
    actual = []
    for prediction in predict_results:
        predicted.append(prediction['prediction'][0])
        actual.append(prediction['y'][0])
    print("Predict :",predicted[:10])
    print("Actual :",actual[:10])

train()
