import argparse
import tensorflow as tf
from tensorflow_core.python.lib.io import file_io

#Disable eager execution
tf.compat.v1.disable_eager_execution()

#Instantiate the distribution strategy -- ParameterServerStrategy. This needs to be in the beginning of the code.
strategy = tf.distribute.experimental.ParameterServerStrategy()

#Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
      "--input_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be cloud storage"
)
parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be cloud storage"
)
FLAGS, unparsed = parser.parse_known_args()

# Load MNIST data using built-in datasets' download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

BUFFER_SIZE = len(x_train)
BATCH_SIZE_PER_REPLICA = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
EPOCHS = 10
STEPS_PER_EPOCH = int(BUFFER_SIZE/EPOCHS)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE,drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE)


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

#Save checkpoints to the output location -- most probably on a cloud storage, such as GCS
callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.output_path)
# Finally, train or fit the model
history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[callback], use_multiprocessing=True)
# history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,  use_multiprocessing=True)
# Save the model to the cloud storage
model.save("model.h5")
with file_io.FileIO('model.h5', mode='r') as input_f:
    with file_io.FileIO(FLAGS.output_path+ '/model.h5', mode='wb+') as output_f:
        output_f.write(input_f.read())

