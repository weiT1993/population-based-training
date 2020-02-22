import tensorflow as tf
import numpy as np
from utils.datasets import get_dataset


(x_train, y_train), (x_test, y_test) = get_dataset(data_file='./data/power_15freq_7.3202.mat',time_range=300,concat=False)

print(type(x_train))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, 300)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])

predictions = model(x_train[:1]).numpy()
predictions = tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], predictions).numpy()

print(predictions,np.sum(predictions))
print('no training loss = %.3f'%loss)

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)