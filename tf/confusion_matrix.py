from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)
        
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    return figure

QTDE_TEST = 348

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(flow, QTDE_TEST // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)

print('confusion matrix')
cm = confusion_matrix(flow.classes, y_pred)
print(cm)

print('classification report')
target_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
print(classification_report(flow.classes, y_pred, target_names=target_names))

#fig = plot_confusion_matrix(cm, target_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

fig = disp.plot().figure_
fig.savefig('confusion_matrix.png')
