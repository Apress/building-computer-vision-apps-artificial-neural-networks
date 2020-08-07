import pathlib
import  cv2
import tensorflow as tf
from tensorflow_core._api.v2.io import gfile
from tensorflow_core.core.framework.graph_pb2 import GraphDef
from tensorflow_core.python.framework.graph_io import write_graph

print(tf.version.VERSION)
model_path="/Users/sansari/Downloads/20180408-102900/"
# # Model preparation
def load_model(model_path):
    model_dir = pathlib.Path(model_path)
    model = tf.keras.models.load_model(str(model_dir))
    print(model.signatures)
   # model = model.signatures['serving_default']
    return model

model = load_model(model_path)


def convert_pb_to_pbtxt(filename):
  with gfile.GFile(filename,'rb') as f:
    graph_def = GraphDef()

    graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

    write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
  return

# convert_pb_to_pbtxt("/Users/sansari/Downloads/20180408-102900/20180408-102900.pb")
image = cv2.imread("/Users/sansari/Downloads/test_images/catotherpets.jpg")
# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# Run prediction from the model
output_dict = model(input_tensor)
print(output_dict)