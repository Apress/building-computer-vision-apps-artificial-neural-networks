import argparse

import tensorflow as tf
# import keras

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

parser = argparse.ArgumentParser()
parser.add_argument(
      "--input_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be clous storage"
)
parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be clous storage"
)

FLAGS, unparsed = parser.parse_known_args()


tf.print("TF version::",  tf.version.VERSION)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.output_path)



# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

BUFFER_SIZE = len(x_train)

BATCH_SIZE_PER_REPLICA = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
print("---------- global batch size------------", GLOBAL_BATCH_SIZE)
EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():

    # Build the ANN with 4-layers
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

    # Compile the model and set optimizer,loss function and metrics
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Finally, train or fot the model
history = model.fit(train_dataset, epochs=5, steps_per_epoch=100, callbacks=[callback])
model.save("model1.h5")
# print(history)
# # Evaluate the result using the test set.\
# evalResult = model.evaluate(x_test,  y_test, verbose=1)
# print("Evaluation", evalResult)
# predicted = model.predict(x_test)
# print("Predicted", predicted)
