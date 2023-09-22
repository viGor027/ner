import tensorflow as tf
from data import X_train, y_train, X_valid, y_valid, categories, n_words
from kerastuner.tuners import BayesianOptimization
from custom_scc import custom_scc


def build_model(hp):
    model_rnn = tf.keras.Sequential()
    model_rnn.add(tf.keras.layers.GRU(units=hp.Int('units_0', min_value=32, max_value=512, step=32),
                                      input_shape=(None, n_words), return_sequences=True))

    for i in range(1, hp.Int('num_layers', 0, 4) + 1):
        model_rnn.add(
            tf.keras.layers.GRU(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                return_sequences=True)
        )

    model_rnn.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(categories, activation="softmax"))
    )

    model_rnn.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=custom_scc)
    return model_rnn


es = tf.keras.callbacks.EarlyStopping(patience=2)
tb = tf.keras.callbacks.TensorBoard('tensorboard/')

metrics = [
    tf.keras.metrics.F1Score(name='F_1')
]

tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    directory='tune',
    project_name='NER'
    )

tuner.search(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=[es], batch_size=256, epochs=10)
best_model = tuner.get_best_models()[0]
best_model.save('model.keras')
best_model.save('model.h5')
