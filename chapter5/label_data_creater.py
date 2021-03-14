import tensorflow as tf
import numpy as np
import os
import pathlib

print(tf.__version__)


def get_ds(data_dir_str, batch_size=32, img_height=224, img_width=224):
    data_dir = pathlib.Path(data_dir_str)
    image_count = len(list(data_dir.glob('*/*')))
    STEPS_PER_EPOCH = np.ceil(image_count / batch_size)
    # compute the class name from the dir
    class_names = np.array([item.name for item in data_dir.glob('*')])

    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)

        # The second to last is the class-directory

        return parts[-2] == class_names

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [img_width, img_height])

    def process_path(file_path):
        label = get_label(file_path)

        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    train_ds = prepare_for_training(labeled_ds, cache=True)

    image_batch, label_batch = next(iter(train_ds))

    return image_batch.numpy(), label_batch.numpy()
