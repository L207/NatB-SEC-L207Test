# Execution Instructions
# Set Parameters as follows:
#

# TensorFlow and tf.keras
import keras.optimizers
import tensorflow as tf

# Helper libraries
import numpy as np
import data_store
import input_data
import sys

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.4f}'.format})

print(len(sys.argv))
print(sys.argv)
print(sys.path[0])

#cl_weightarr = float(sys.argv[1:5])
cl_weightarr = [float(x) for x in sys.argv[1:5]]
lr = float(sys.argv[5])
lay1w = int(sys.argv[6])

filename = "D:/Users/Luke/Documents/SEChallenge2022/DataSet/SECH.IncrementalDelayAndLoss_Eth1"

data_store.init()

print(tf.__version__)

input_data.get_inputs_and_outputs(filename)
#https://www.tensorflow.org/tutorials/structured_data/imbalanced_data (Dropout Layer)
#32 length layers worked very well ... 61.7% accurate validation on one occasion
#however a re-run attempt shown this was pure luck - only 4%.
#output_bias = np.array([0.03, 0.9, 0.9, 0.9])
#output_bias = keras.initializers.Constant([0.1, 0.1, 10, 0.1])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(len(data_store.input_d[0])),
    tf.keras.layers.Dense(lay1w, activation='elu'),
    #tf.keras.layers.Dense(8, activation='sigmoid'),
    #tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(8, activation='softmax'),
    #tf.keras.layers.Dense(len(data_store.output_d[0]), activation='softmax', use_bias=True, bias_initializer=output_bias)
    tf.keras.layers.Dense(len(data_store.output_d[0]), activation='softmax', use_bias=True)
])

#model.layers[-1].bias.assign([0.03,0.9,0.9,0.9])
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

if sys.argv[7] == "BinCross":
    loss_tp = tf.keras.losses.BinaryCrossentropy(from_logits=False)
elif sys.argv[7] == "MeanSq":
    loss_tp = tf.keras.losses.MeanSquaredError()
elif sys.argv[7] == "MeanAbs":
    loss_tp = tf.keras.losses.MeanAbsoluteError()
elif sys.argv[7] == "CatCross":
    loss_tp = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
else:
    loss_tp = tf.keras.losses.BinaryCrossentropy(from_logits=False)

opt = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer=opt, #optimizer='adam',
              #loss=tf.keras.losses.MeanSquaredError(),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              loss=loss_tp,
              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              #loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['accuracy'])


#model.train(data_store.input_d, data_store.output_d, epochs=10)

#cl_weight = {0: 0.5, 1: 0.001, 2: 2.0, 3: 0.001}
cl_weight = {0: cl_weightarr[0],
             1: cl_weightarr[1],
             2: cl_weightarr[2],
             3: cl_weightarr[3]}

print("Starting Training Process")
#model.fit(data_store.input_d, data_store.output_d, verbose=2, epochs=30, validation_split=0.4)
model.fit(data_store.input_d, data_store.output_d, verbose=2, epochs=60, validation_split=0.4, class_weight=cl_weight)
print("Training Step Complete")
_, accuracy = model.evaluate(data_store.input_d, data_store.output_d)
print("Accuracy : " + str(accuracy))
predict_d = model.predict(data_store.input_d)
#for i in range(0, len(data_store.output_d)):
#    print(str(i) + " : " + str(predict_d[i]) + " : " + str(data_store.output_d[i]))

wr_array = np.hstack((predict_d, data_store.output_d))
accstr = "{:.4f}".format(accuracy)

np.savetxt((sys.path[0] + "\Out-" + accstr + "_" +
            sys.argv[1] + "-" + sys.argv[2] + "-" + sys.argv[3] + "-" + sys.argv[4] + "-" +
            sys.argv[5] + "-" + sys.argv[6] + "-" + sys.argv[7] + ".csv"), wr_array)

print("program complete.")

exit()
#use write and restore to get the trained model saved and restored for operation?
#see https://www.tensorflow.org/guide/intro_to_modules