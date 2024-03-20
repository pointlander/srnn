import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(xTrain , yTrain) , (xTest , yTest) = fashion_mnist.load_data()

xTrain = xTrain/255
yTrain = yTrain/255
xTest = xTest/255 
yTest = yTest/255 

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 128 , activation = "leaky_relu"),
    tf.keras.layers.Dense(units = 10 , activation = tf.keras.activations.softmax)
])

model.compile(optimizer = "adam" , loss = tf.keras.losses.sparse_categorical_crossentropy)

model.fit(xTrain , yTrain , epochs= 5)

model.evaluate(xTest , yTest)