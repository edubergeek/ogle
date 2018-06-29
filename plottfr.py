import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

data_path = './ogle/data/train.tfr'  # address to save the hdf5 file
with tf.Session() as sess:
    feature = {'train/data': tf.FixedLenFeature([1200], tf.float32),
               'train/period': tf.FixedLenFeature([], tf.float32)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the data from string back to the numbers
    x = features['train/data']
    # reshape into 2 dim ?, 3
    #x = tf.reshape(x, [-1,3])
    
    # Cast label data into int32
    y = features['train/period']
    #y = tf.cast(features['train/period'], tf.float32)
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    X, Y = tf.train.shuffle_batch([x, y], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Now we read batches of images and labels and plot them
    for batch_index in range(3):
        lightcurve, period = sess.run([X, Y])

        lc = np.reshape(lightcurve,(-1, 400, 3))
        for n in range(399,0,-1):
            if lc[0,n,0] > 0.0:
                break
        steps=n+1
        #for n in range(0,399,1):
        #    if lc[0,n,0] - lc[0,0,0] > 30.0:
        #        break
        #steps=n
        Range = lc[0,0:steps,0]
        Series = lc[0,0:steps,1]
        print(period[0])
        plt.plot(Range, Series, 'bo', markersize=2)
        #plt.plot(lightcurve[0], lightcurve[2], 'b')
        #plt.title(lightcurve)
        plt.show()


    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()

