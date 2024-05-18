from keras._tf_keras.keras.utils import load_img, img_to_array
from os import listdir
from pickle import load
import numpy as np
import random
import pyttsx3

test_path = 'ASL_Dataset_Reduced/test/'
text_translation_path = 'Translations/translation_text.txt'
audio_translation_path = 'Translations/translation_audio.mp3'
labels = listdir(test_path)
model = load(open('Models/model_reduced_acc_auc_10.sav', 'rb'))
class_list = ['D', 'E', 'E', 'P', 'Space', 'L', 'E', 'A', 'R', 'N', 'I', 'N', 'G', 'Space', 'R', 'O', 'C', 'K', 'S']
pred_class_list = []
sentence = ''

# Predict random images using provided classes
for i in class_list:
   files = listdir(test_path + i + '/')
   file = random.choice(files)

   image = load_img(test_path + i + '/' + file, target_size=(128, 128))
   image = img_to_array(image)
   image = image.reshape(1, 128, 128, 3)
   image = image * (1.0/255.0)

   pred = model.predict(image)
   classes = np.argmax(pred, axis=1)

   for i in classes:
      pred_class_list.append(labels[i])

# Print actual vs. predicted
print('Actual:')
print(class_list)
print('Predicted:')
print(pred_class_list)

# Construct sentence string
for i in pred_class_list:
   if i == 'Space' or i == 'Nothing':
      letter = ' '
   else:
      letter = i
   sentence += letter
sentence = sentence.capitalize()

# Print translation
print("(Text) Translation:")
print(sentence)

# Save text translation
translation_text_file = open(text_translation_path, 'w')
translation_text_file.write(sentence)
translation_text_file.close()

print("(Audio) Translation:")
engine = pyttsx3.init()
engine.say(sentence)
engine.save_to_file(sentence, audio_translation_path)
engine.runAndWait()