import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import random

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')

flags.DEFINE_string('annotations_dir', 'annotations',
                   '(Relative) path to annotations directory.')
flags.DEFINE_string('image_dir', 'images',
                   '(Relative) path to images directory.')

flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                   'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                    'difficult instances')
FLAGS = flags.FLAGS

# This function generates a list of images for training and validation.
def create_trainval_list(data_dir):
   trainval_filename = os.path.abspath(os.path.join(data_dir,"trainval.txt"))
   trainval = open(os.path.abspath(trainval_filename), "w")
   files = os.listdir(os.path.join(data_dir, FLAGS.image_dir))
   for f in files:
       absfile =os.path.abspath(os.path.join(data_dir, FLAGS.image_dir, f))
       trainval.write(absfile+"\n")
       print(absfile)
   trainval.close()


def dict_to_tf_example(data,
                      dataset_directory,
                      label_map_dict,
                      ignore_difficult_instances=False,
                      image_subdirectory=FLAGS.image_dir):
 """Convert XML derived dict to tf.Example proto.

 Notice that this function normalizes the bounding box coordinates provided
 by the raw data.

 Args:
   data: dict holding PASCAL XML fields for a single image
   dataset_directory: Path to root directory holding PASCAL dataset
   label_map_dict: A map from string label names to integers ids.
   ignore_difficult_instances: Whether to skip difficult instances in the
     dataset  (default: False).
   image_subdirectory: String specifying subdirectory within the
     PASCAL dataset directory holding the actual image data.

 Returns:
   example: The converted tf.Example.

 Raises:
   ValueError: if the image pointed to by data['filename'] is not a valid JPEG
 """
 filename = data['filename']

 if filename.find(".jpg") < 0:
     filename = filename+".jpg"
 img_path = os.path.join("",image_subdirectory, filename)
 full_path = os.path.join(dataset_directory, img_path)

 with tf.gfile.GFile(full_path, 'rb') as fid:
   encoded_jpg = fid.read()
 encoded_jpg_io = io.BytesIO(encoded_jpg)
 image = PIL.Image.open(encoded_jpg_io)
 if image.format != 'JPEG':
   raise ValueError('Image format not JPEG')
 key = hashlib.sha256(encoded_jpg).hexdigest()

 width = int(data['size']['width'])
 height = int(data['size']['height'])

 xmin = []
 ymin = []
 xmax = []
 ymax = []
 classes = []
 classes_text = []
 truncated = []
 poses = []
 difficult_obj = []
 if 'object' in data:
   for obj in data['object']:
     difficult = bool(int(obj['difficult']))
     if ignore_difficult_instances and difficult:
       continue

     difficult_obj.append(int(difficult))

     xmin.append(float(obj['bndbox']['xmin']) / width)
     ymin.append(float(obj['bndbox']['ymin']) / height)
     xmax.append(float(obj['bndbox']['xmax']) / width)
     ymax.append(float(obj['bndbox']['ymax']) / height)
     classes_text.append(obj['name'].encode('utf8'))
     classes.append(label_map_dict[obj['name']])
     truncated.append(int(obj['truncated']))
     poses.append(obj['pose'].encode('utf8'))

 example = tf.train.Example(features=tf.train.Features(feature={
     'image/height': dataset_util.int64_feature(height),
     'image/width': dataset_util.int64_feature(width),
     'image/filename': dataset_util.bytes_feature(
         data['filename'].encode('utf8')),
     'image/source_id': dataset_util.bytes_feature(
         data['filename'].encode('utf8')),
     'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
     'image/encoded': dataset_util.bytes_feature(encoded_jpg),
     'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
     'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
     'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
     'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
     'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
     'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
     'image/object/class/label': dataset_util.int64_list_feature(classes),
     'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
     'image/object/truncated': dataset_util.int64_list_feature(truncated),
     'image/object/view': dataset_util.bytes_list_feature(poses),
 }))
 return example

def create_tf(examples_list, annotations_dir, label_map_dict, dataset_type):
   writer = None
   if not os.path.exists(FLAGS.output_path+"/"+dataset_type):
       os.mkdir(FLAGS.output_path+"/"+dataset_type)

   j = 0
   for idx, example in enumerate(examples_list):

       if idx % 100 == 0:
           logging.info('On image %d of %d', idx, len(examples_list))
           print((FLAGS.output_path + "/tf_training_" + str(j) + ".record"))
           writer = tf.python_io.TFRecordWriter(FLAGS.output_path + "/"+dataset_type+"/tf_training_" + str(j) + ".record")
           j = j + 1

       path = os.path.join(annotations_dir, os.path.basename(example).replace(".jpg", '.xml'))

       with tf.gfile.GFile(path, 'r') as fid:
           xml_str = fid.read()
       xml = etree.fromstring(xml_str)
       data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

       tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                       FLAGS.ignore_difficult_instances)
       writer.write(tf_example.SerializeToString())

def main(_):

   data_dir = FLAGS.data_dir
   create_trainval_list(data_dir)

   label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

   examples_path = os.path.join(data_dir,'trainval.txt')
   annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
   examples_list = dataset_util.read_examples_list(examples_path)

   random.seed(42)
   random.shuffle(examples_list)
   num_examples = len(examples_list)
   num_train = int(0.7 * num_examples)
   train_examples = examples_list[:num_train]
   val_examples = examples_list[num_train:]

   create_tf(train_examples, annotations_dir, label_map_dict, "train")
   create_tf(val_examples, annotations_dir, label_map_dict, "val")

if __name__ == '__main__':
   tf.app.run()
