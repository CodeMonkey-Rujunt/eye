import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os

batch_size = 16
learning_rate = 1e-4
ceh = False
focalloss = False

# create a CLAHE with L channel(Contrast Limited Adaptive Histogram Equalization)
def CEH2(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl1

# create a CLAHE (Contrast Limited Adaptive Histogram Equalization)
def CEH(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize)
    cl1 = clahe.apply(img_bw)
    return cl1

# create Equalization Histogram
def EH(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_bw)
    eh1 = np.hstack((img_bw, equ))
    return eh1

# reduce the black background
def cut_img(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    img_cut = img[y:y+h, x:x+w]
    return img_cut

# reduce the black background
def cut_and_resize_to_original_img(img):
    shp = img.shape[0:2]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    img_cut = img[y:y+h, x:x+w]
    bgr_final = cv2.cvtColor(img_cut, cv2.COLOR_LAB2BGR)
    img_cut_resized = cv2.resize(bgr_final, shp, interpolation=cv2.INTER_AREA)
    return img_cut_resized

def CEH_cut_pipeline(img):
    img_uint = img.astype(np.uint8)
    img1 = cut_and_resize_to_original_img(img_uint)
    img2 = CEH2(img1)
    return img2

def load_images_from_folder(path_folder):
    PROC_FOLDER = path_folder + '_procEH/'
    if os.path.isdir(os.path.dirname(PROC_FOLDER)) is False:
        os.makedirs(os.path.dirname(PROC_FOLDER))

    for filename in os.listdir(path_folder):
        img = cv2.imread(os.path.join(path_folder, filename))
        if img is not None:
            img_proc = cut_img(img)
            img_proc = EH(img) # change with EH
            path = os.path.join(PROC_FOLDER, filename)
            cv2.imwrite(path, img_proc)

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
	
base_model = EfficientNetB3(weights='imagenet')
out = base_model.get_layer('top_dropout').output

#base_model = VGG16(weights='imagenet')
#out = base_model.get_layer('fc2').output

#base_model = InceptionResNetV2(weights='imagenet')
#out = base_model.get_layer('avg_pool').output

out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(base_model.input, out)

# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['AUC'])
#model.compile(optimizer='nadam', loss=focal_loss(gamma=2.0, alpha=0.2), metrics=['AUC'])

datagen = ImageDataGenerator(validation_split=0.2)
#datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocessing.CEH_cut_pipeline)
