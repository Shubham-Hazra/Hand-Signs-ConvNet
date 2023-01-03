import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras

# Load the data
train_dataset = h5py.File('train_signs.h5', "r")
x_train = np.array(train_dataset["train_set_x"][:])
y_train = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('test_signs.h5', "r")
x_test = np.array(test_dataset["test_set_x"][:])
y_test = np.array(test_dataset["test_set_y"][:])

# Normalize image vectors
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape
y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

# One-hot encoding
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.fit_transform(y_test)
classes = y_train.shape[1]

# Define the model


def model(input_shape):
    image = tf.keras.Input(input_shape)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.ReLU()(batch1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(relu1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    batch2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.layers.ReLU()(batch2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(relu2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(pool2)
    batch3 = tf.keras.layers.BatchNormalization()(conv3)
    relu3 = tf.keras.layers.ReLU()(batch3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(relu3)
    flat = tf.keras.layers.Flatten()(pool3)
    dense1 = tf.keras.layers.Dense(128)(flat)
    relu4 = tf.keras.layers.ReLU()(dense1)
    dense2 = tf.keras.layers.Dense(64)(relu4)
    relu5 = tf.keras.layers.ReLU()(dense2)
    dense3 = tf.keras.layers.Dense(classes, activation='softmax')(relu5)
    model = tf.keras.Model(inputs=image, outputs=dense3)
    return model


model = model(x_train.shape[1:])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=16,
          validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

# Print the results
print('Test accuracy:', test_acc*100, '%')

# Save the model
model.save('signs.h5')

# Load the model
model = keras.models.load_model('signs.h5')

# Predict
random_image = np.random.randint(0, x_test.shape[0])
plt.imshow(x_test[random_image])
prediction = np.argmax(model.predict(
    x_test[random_image].reshape(1, 64, 64, 3), verbose=0))
print('Prediction:', prediction)
plt.show()
