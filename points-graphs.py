from __future__ import print_function
import warnings
import logging, os
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

# Define el modelo de red neuronal
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Dense_2_4'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Dense_4_8'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Dense_8_1'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output')
])

# Compila el modelo
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Entrena el modelo
# (Se omite el código de entrenamiento para simplificar el ejemplo)

# Guarda el modelo utilizando el método de graphs
writer = tf.summary.create_file_writer('./map-model')

with writer.as_default():
    graph = tf.function(linear_model).get_concrete_function(tf.TensorSpec(shape=[None, 2], dtype=tf.float32))
    summary_ops_v2.graph(graph.graph.as_graph_def())

writer.close()
