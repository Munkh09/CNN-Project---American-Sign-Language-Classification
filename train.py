from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from os import listdir
from pickle import dump

train_path = 'ASL_Dataset_Reduced/train'
val_path = 'ASL_Dataset_Reduced/val'
labels = listdir(train_path)

# Train set
# Normalize pixel values to between 0, 1
# Allow horizontal flip
# Set shear transformation range
# Set brightness range
# Allow rotation
# Allow width, height shift
# Set zoom range
# Set fill mode to 'nearest'
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, shear_range=0.1, brightness_range=[0.8, 1.2], rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest')

# Val set
# Normalize pixel values to between 0, 1
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(directory=train_path, target_size=(128, 128), classes=labels, batch_size=32, class_mode='categorical', seed=1337)
val_gen = val_datagen.flow_from_directory(directory=val_path, target_size=(128, 128), classes=labels, batch_size=32, class_mode='categorical', seed=1337)

# Build model
# Architecture influenced by VGG-Net, which performs well for classification
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(28, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
callbacks = [ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=0.0001, verbose=1)]
history = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks, verbose=1)

# Save model
dump(model, open('Models/model_reduced_acc_auc_10.sav', 'wb'))
dump(history.history, open('History/model_reduced_acc_auc_10_history', 'wb'))