from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3, InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

BATCH_SIZE = 16
EPOCHS = 20
SEED = 13
LR = 1e-4
ceh = False
focalloss = False
efficientnet = True

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        # Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
        epsilon = 1e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed
	
if efficientnet:
    base_model = EfficientNetB3(weights='imagenet')
elif vgg16:
    base_model = VGG16(weights='imagenet')
else:
    base_model = InceptionResNetV2(weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

if efficientnet:
    out = base_model.get_layer('top_dropout').output
elif vgg16:
    out = vgg_conv.get_layer('fc2').output
else:
    out = base_model.get_layer('avg_pool').output

out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(base_model.input, out)

# We compile the model
if focalloss:
    model.compile(optimizer='nadam', loss=focal_loss(gamma=2.0, alpha=0.2), metrics=['AUC'])
else:
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['AUC'])

if ceh:
    datagen = ImageDataGenerator(validation_split=0.2)
else:
    datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocessing.CEH_cut_pipeline)

train_gen = datagen.flow_from_directory('/work/ocular-dataset/ODIR-5K-Flow/train/',
        target_size=(244,244), batch_size=BATCH_SIZE, subset='training', seed=SEED)

val_gen = datagen.flow_from_directory('/work/ocular-dataset/ODIR-5K-Flow/train/',
        target_size=(244,244), batch_size=BATCH_SIZE, subset='validation', seed=SEED)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint(EXP_NAME + '_best_model.h5', monitor='val_auc', mode='max', verbose=1, save_best_only=True)

# fine-tune the model
history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.n // BATCH_SIZE,
        callbacks=[es, mc])

loss_values = history.history['loss']
#loss_values = history.history['auc']
loss_val_values = history.history['val_loss']
epochs = range(1, len(loss_values)+1)

fig, ax = plt.subplots()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.plot(epochs, loss_values, '-o', label='Training Loss')
plt.plot(epochs, loss_val_values, '-o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epochs')
plt.legend()

textstr = 'best val_auc: ' + str(round(max(history.history['val_auc']),4))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

plt.savefig('result.png')
