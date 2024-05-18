from keras._tf_keras.keras.applications import VGG16, InceptionV3
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from keras._tf_keras.keras.models import Model
from os import listdir
from pickle import dump

train_path = 'ASL_Dataset_Reduced/train'
val_path = 'ASL_Dataset_Reduced/val'
labels = listdir(train_path)

# Train set
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, shear_range=0.1, brightness_range=[0.8, 1.2], rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest')

# Val set
# Normalize pixel values to between 0, 1
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(directory=train_path, target_size=(128, 128), classes=labels, batch_size=32, class_mode='categorical', seed=1337)
val_gen = val_datagen.flow_from_directory(directory=val_path, target_size=(128, 128), classes=labels, batch_size=32, class_mode='categorical', seed=1337)

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

callbacks = [ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=0.0001, verbose=1)]

model = Model(inputs=vgg16.inputs, outputs=vgg16_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
history = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[callbacks], verbose=1)

# Save vgg16
dump(model, open('Models/vgg16_reduced_acc_auc_10.sav', 'wb'))
dump(history.history, open('History/vgg16_reduced_acc_auc_10_history', 'wb'))

model = Model(inputs=inceptionv3.inputs, outputs=inceptionv3_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
history = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[callbacks], verbose=1)

# Save inceptionv3
dump(model, open('Models/inceptionv3_reduced_acc_auc_10.sav', 'wb'))
dump(history.history, open('History/inceptionv3_reduced_acc_auc_10_history', 'wb'))