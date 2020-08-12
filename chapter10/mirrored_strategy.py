import tensorflow as tf

# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

BUFFER_SIZE = len(x_train)
BATCH_SIZE_PER_REPLICA = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
EPOCHS = 100
STEPS_PER_EPOCH = int(BUFFER_SIZE/EPOCHS)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE,drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
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
history = model.fit(test_dataset, epochs=EPOCHS)

# Evaluate the result using the test set.\
evalResult = model.evaluate(x_test,  y_test, verbose=1)
print("Evaluation", evalResult)
predicted = model.predict(x_test)
print("Predicted", predicted)
