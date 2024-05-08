from keras._tf_keras.keras.utils import load_img, img_to_array
from os import listdir
from pickle import load
import numpy as np
import random
import pyttsx3

test_path = 'ASL_Dataset_165k/test/'
labels = listdir(test_path)
model = load(open('Models/model_165k_acc_auc_20.sav', 'rb'))
class_list = ['H', 'E', 'L', 'L', 'O', 'Space', 'W', 'O', 'R', 'L', 'D']
pred_class_list = []
sentence = ''

for i in class_list:
   files = listdir(test_path + i + '/')
   file = random.choice(files)

   image = load_img(test_path + i + '/' + file, target_size=(200, 200))
   image = img_to_array(image)
   image = image.reshape(1, 200, 200, 3)
   image = image * (1.0/255.0)

   pred = model.predict(image)
   classes = np.argmax(pred, axis=1)

   for i in classes:
      pred_class_list.append(labels[i])

print('Actual:')
print(class_list)
print('Predicted:')
print(pred_class_list)

for i in pred_class_list:
   if i == 'Space' or i == 'Nothing':
      letter = ' '
   else:
      letter = i
   sentence += letter

print("(Text) Translation:")
print(sentence.capitalize())

print("(Audio) Translation:")
engine = pyttsx3.init()
engine.say(sentence)
engine.runAndWait()