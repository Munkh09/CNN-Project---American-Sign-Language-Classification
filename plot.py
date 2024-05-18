import matplotlib.pyplot as plt
from pickle import load

# Replace file path with necessary history for each model
history = load(open('History/model_reduced_acc_auc_10_history', 'rb'))

# Accuracy
plt.plot(history['accuracy'], label='Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# AUC
plt.plot(history['auc'], label='AUC')
plt.plot(history['val_auc'], label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC vs. Validation AUC')
plt.legend(loc='lower right')
plt.show()

# Loss
plt.plot(history['loss'], label='Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Validation Loss')
plt.legend(loc='upper right')
plt.show()