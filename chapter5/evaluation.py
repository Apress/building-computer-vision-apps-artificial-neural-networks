import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.executing_eagerly()

# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values by deciding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the ANN with 4-layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model and set optimizer,loss function and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Finally, train or fot the model
history = model.fit(x_train, y_train, validation_split=0.3, epochs=10)

# Visualize loss  and accuracy history
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['accuracy'], 'b-')
plt.legend(['Training Loss', 'Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Percent')
# plt.show();

# Evaluate the result using the test set.\
evalResult = model.evaluate(x_test, y_test, verbose=1)
print("Evaluation", evalResult)
predicted = model.predict(x_test)
print("Predicted", predicted)
print("prediction class", np.argmax(predicted, axis=1))

confusion = tf.math.confusion_matrix(y_test, np.argmax(predicted, axis=1), num_classes=10)

tf.print(confusion)
