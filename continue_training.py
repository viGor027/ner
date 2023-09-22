import tensorflow as tf
from data import X_train, y_train, X_valid, y_valid, categories, n_words
from custom_scc import custom_scc


es = tf.keras.callbacks.EarlyStopping(patience=2)
tb = tf.keras.callbacks.TensorBoard('tensorboard/')

best_model = tf.keras.models.load_model('tune/model.h5', custom_objects={'custom_scc': custom_scc})
best_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=[es, tb], batch_size=256, epochs=100)

best_model.save('model_final.keras')
best_model.save('model_final.h5')
