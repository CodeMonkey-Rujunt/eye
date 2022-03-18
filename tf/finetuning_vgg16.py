from tensorflow.keras.applications.vgg16 import VGG16

EXP_NAME = 'ft_vgg16_fc2_lr-4'

vgg_conv = VGG16(weights='imagenet')

out = vgg_conv.get_layer('fc2').output
