import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# numpyでテストデータを作成
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

#model.fit(data, labels, epochs=2, batch_size=32)
model.evaluate(data, labels, batch_size=32)
#model.evaluate(dataset, steps=30)

print("****")
result = model.predict(data, batch_size=32)
print(result.shape)
