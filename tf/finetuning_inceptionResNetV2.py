from tensorflow.keras.applications import InceptionResNetV2

EXP_NAME = "ft_InceptionResNetV2_avg_pool_lr-4"

base_model = InceptionResNetV2(weights='imagenet')

out = base_model.get_layer('avg_pool').output
