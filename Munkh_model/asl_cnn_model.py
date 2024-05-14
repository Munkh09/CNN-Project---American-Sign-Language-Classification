# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:58:18 2024

@author: Munkh
"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras
import keras.backend
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py

# Define data generators for training and validation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    rescale=1./255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/khmun/Downloads/CS171_CNN_Project/Train',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/khmun/Downloads/CS171_CNN_Project/Test',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Do not shuffle the images, so they remain in the order you specify
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/khmun/Downloads/CS171_CNN_Project/Validation',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3), strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.250))

model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.250))

model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.250))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.250))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.250))
model.add(Dense(28, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=2, mode='max', verbose=1),
    ModelCheckpoint('best_cnn_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
]
history = model.fit(train_generator, epochs=5, validation_data=validation_generator, callbacks=[callbacks], verbose=1)


loss, accuracy = model.evaluate(test_generator, verbose=0)
print("------------------------------------------")
print(f"Loss on test data: {loss}")
print(f"Accuracy on test data: {accuracy}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
