from keras._tf_keras.keras.applications import VGG16, InceptionV3
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from os import listdir

test_path = 'ASL_Dataset_Train_Oversampled/test/'
labels = listdir(test_path)

# Normalize pixel values to between 0, 1
test_datagen = ImageDataGenerator(rescale=(1.0/255.0))

test_gen = test_datagen.flow_from_directory(directory=test_path, target_size=(128, 128), classes=labels, class_mode='categorical', seed=1337)

# VGG16
vgg16 = VGG16(weights='imagenet', input_shape=(128, 128, 3), classes=28, include_top=False)

for layer in vgg16.layers:
   layer.trainable = False

x = Flatten()(vgg16.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
vgg16_output = Dense(28, activation='softmax', name='predictions')(x)

# InceptionV3
inceptionv3 = InceptionV3(weights='imagenet', input_shape=(128, 128, 3), classes=28, include_top=False)

for layer in inceptionv3.layers:
   layer.trainable = False

x = GlobalAveragePooling2D(name='avg_pool')(inceptionv3.output)
inceptionv3_output = Dense(28, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.inputs, outputs=vgg16_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
model.evaluate(test_gen)

model = Model(inputs=inceptionv3.inputs, outputs=inceptionv3_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
model.evaluate(test_gen)