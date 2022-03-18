from tensorflow.keras.applications.vgg16 import VGG16
import preprocessing

EXP_NAME = "ft_vgg16_fc2_lr-4_ceh_cut"
PRE_PROC_FUNC = pre_processing.CEH_cut_pipeline
vgg_conv = VGG16(weights='imagenet')
out = vgg_conv.get_layer('fc2').output
datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=PRE_PROC_FUNC)
