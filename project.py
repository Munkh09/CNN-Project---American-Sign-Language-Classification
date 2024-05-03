from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from os import listdir

train_path = 'ASL_Dataset_13k/train'
val_path = 'ASL_Dataset_13k/val'
labels = listdir(train_path)

train = image_dataset_from_directory(train_path)
test = image_dataset_from_directory(val_path)

# Augment images for more robust model

# Train set
# Normalize pixel values to 0, 1
# Flip horizontally (hand signs can be used by either hand, but not upside down)
# Rotate by 45 degrees either side
# Shift left, right, up, down
# Zoom in, out
# train_datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, rescale=1.0/255.0)
train_datagen = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True, rescale=1.0/255.0)

# Val set
# Normalize pixel values to 0, 1
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(directory=train_path, target_size=(400,400), classes=labels, class_mode='categorical', batch_size=32, seed=1337)
val_gen = val_datagen.flow_from_directory(directory=val_path, target_size=(400,400), classes=labels, class_mode='categorical', batch_size=32, seed=1337)

print(train_gen.class_indices)

# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(28 , activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=12)