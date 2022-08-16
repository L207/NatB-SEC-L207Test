# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import data_store
import input_data

filename = "D:/Users/Luke/Documents/SEChallenge2022/DataSet/SECH.IncrementalDelayAndLoss_Eth1"

data_store.init()

print(tf.__version__)

input_data.get_inputs_and_outputs(filename)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(len(data_store.input_d[0])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(data_store.output_d[0]))
])

#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

#model.train(data_store.input_d, data_store.output_d, epochs=10)

print("Starting Training Process")
model.fit(data_store.input_d, data_store.output_d, verbose=2, epochs=10)
print("Training Step Complete")
_, accuracy = model.evaluate(data_store.input_d, data_store.output_d)
print("Accuracy : " + str(accuracy))
print("program complete.")