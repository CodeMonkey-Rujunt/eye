from tensorflow.keras.applications import EfficientNetB3

def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        # Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed
	
EXP_NAME = "ft_efficientnetb3_top_dropout_lr-4"

# We compile the model
model.compile(optimizer='nadam', loss=focal_loss(gamma = 2.0, alpha=0.2), metrics=['AUC'])

loss_values = history.history['auc']
