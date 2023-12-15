import numpy as np
import os
import pandas as pd
import tensorflow as tf

def circulo(num_datos=740000, R=1, centro_lat=0, centro_lon=0):
    pi = np.pi
    theta = np.random.uniform(0, 2 * pi, size=num_datos)

    r_positive = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    x = np.cos(theta) * r_positive + centro_lon
    y = np.sin(theta) * r_positive + centro_lat

    x = np.round(x, 6)
    y = np.round(y, 6)

    df = pd.DataFrame({'lat': y, 'lon': x})
    return df

datos_la = circulo(num_datos=100, R=2, centro_lat=34.05223, centro_lon=-118.24368)
datos_paris = circulo(num_datos=100, R=0.5, centro_lat=48.85341, centro_lon=2.3488)

X = np.concatenate([datos_la, datos_paris])
X = np.round(X, 6)
y = np.concatenate([np.zeros(800), np.ones(100), np.ones(100)])

train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()

linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Dense_2_4'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Dense_4_8'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Dense_8_1'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=400)

export_path = 'map-model/1/' 
tf.saved_model.save(linear_model, os.path.join('./', export_path))
