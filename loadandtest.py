# TensorFlow and tf.keras
import keras.optimizers
import tensorflow as tf

# Helper libraries
import numpy as np
import data_store
import input_data
import sys

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})
filename = "D:/Users/Luke/Documents/SEChallenge2022/DataSet/SECH.IncrementalDelayAndLoss_Eth1"
data_store.init()
input_data.get_inputs_and_outputs(filename)
lay1w = 60 #testing only, should load this param.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(len(data_store.input_d[0])),
    tf.keras.layers.Dense(lay1w, activation='elu'),
    #tf.keras.layers.Dense(8, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(15, activation='elu'),
    #tf.keras.layers.Dense(8, activation='softmax'),
    #tf.keras.layers.Dense(len(data_store.output_d[0]), activation='softmax', use_bias=True, bias_initializer=output_bias)
    tf.keras.layers.Dense(len(data_store.output_d[0]), activation='softmax', use_bias=True)
])

model.load_weights("./modelsave.sec")

predict_d = model.predict(data_store.input_d)

wr_array = np.hstack((predict_d, data_store.output_d))

np.savetxt((sys.path[0] + "\Outload_" +
            ".csv"), wr_array, fmt='%0.4f')

print("done")

exit()