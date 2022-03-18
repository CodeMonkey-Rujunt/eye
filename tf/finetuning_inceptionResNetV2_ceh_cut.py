from tensorflow.keras.applications import InceptionResNetV2
import preprocessing

EXP_NAME = "ft_InceptionResNetV2_avg_pool_lr-4_ceh_cut"
PRE_PROC_FUNC = pre_processing.CEH_cut_pipeline
base_model = InceptionResNetV2(weights='imagenet')
out = base_model.get_layer('avg_pool').output
datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=PRE_PROC_FUNC)
