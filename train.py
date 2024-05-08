from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from os import listdir
from pickle import dump

train_path = 'ASL_Dataset_165k/train'
val_path = 'ASL_Dataset_165k/val'
labels = listdir(train_path)

# Train set
# Normalize pixel values to between 0, 1
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Val set
# Normalize pixel values to between 0, 1
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(directory=train_path, target_size=(200,200), classes=labels, class_mode='categorical', batch_size=32, seed=1337)
val_gen = val_datagen.flow_from_directory(directory=val_path, target_size=(200,200), classes=labels, class_mode='categorical', batch_size=32, seed=1337)

# Build model
# Architecture influenced by VGG-Net, which performs well for classification
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'auc'])
model.fit(train_gen, validation_data=val_gen, epochs=20)

# Save model
dump(model, open('Models/model_165k_acc_auc_20.sav', 'wb'))