from tensorflow.keras.applications import EfficientNetB3
import preprocessing

EXP_NAME = 'ft_efficientnetb3_top_dropout_lr-4_ceh_cut'

PRE_PROC_FUNC = pre_processing.CEH_cut_pipeline

base_model = EfficientNetB3(weights='imagenet')

datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=PRE_PROC_FUNC)
