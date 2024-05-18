from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from pickle import load
from os import listdir

test_path = 'ASL_Dataset_Reduced/test/'
labels = listdir(test_path)
model = load(open('Models/model_reduced_acc_auc_10.sav', 'rb'))

# Normalize pixel values to between 0, 1
test_datagen = ImageDataGenerator(rescale=(1.0/255.0))

test_gen = test_datagen.flow_from_directory(directory=test_path, target_size=(128, 128), classes=labels, class_mode='categorical', batch_size=32, seed=1337)

model.evaluate(test_gen)