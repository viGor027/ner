from data import X_test, y_test
import tensorflow as tf
from custom_scc import custom_scc


model = tf.keras.models.load_model('model/model_final.h5', custom_objects={'custom_scc': custom_scc})
model.evaluate(X_test, y_test)
