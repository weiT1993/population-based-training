import tensorflow as tf
import numpy as np
from utils.datasets import get_dataset
from utils.helper_fun import read_file

data_dict = read_file(filename='./data/dataset.p')
x_train, y_train = data_dict['train']
x_valid, y_valid = data_dict['valid']
x_test, y_test = data_dict['test']

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(2, 300),name='input_layer'),
#   tf.keras.layers.Dense(128, activation='relu',name='dense_0'),
#   tf.keras.layers.Dropout(0.1, name='dp'),
#   tf.keras.layers.Dense(2,activation='softmax',name = 'output_layer')
# ])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32,kernel_size=1, activation='relu', input_shape=(2, 300))
  # tf.keras.layers.Flatten(),
  # tf.keras.layers.Dense(2, activation='softmax')
])

print(type(model))
model.summary()

predictions = model(x_train[:1]).numpy()
print(predictions.shape)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss = loss_fn(y_train[:1], predictions).numpy()

# print(predictions,np.sum(predictions))
# print('no training loss = %.3f'%loss)

# model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=1)
# model.evaluate(x_test, y_test, verbose=2)

# model.save('my_model.h5')

# new_model = tf.keras.models.load_model('my_model.h5')
# new_model.summary()
# loss, acc = new_model.evaluate(x_test,  y_test, verbose=2)
# print('Restored model, accuracy: {:5.5f}%'.format(100*acc))