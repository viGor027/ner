import tensorflow as tf


def custom_scc(y_value, y_pred):
    if isinstance(y_value, tf.RaggedTensor):
        y_value = y_value.to_tensor()
    if isinstance(y_pred, tf.RaggedTensor):
        y_pred = y_pred.to_tensor()
    return tf.keras.losses.SparseCategoricalCrossentropy()(y_value, y_pred)