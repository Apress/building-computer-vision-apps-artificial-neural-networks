import os
import pathlib
import random
import numpy as np
import tensorflow as tf
import cv2
# Import the object detection module.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# to make gfile compatible with v2
tf.gfile = tf.io.gfile

model_path = "ssd_model/final_model"
labels_path = "models/research/object_detection/data/pet_label_map.pbtxt"
image_dir = "images"
image_file_pattern = "*.jpg"
output_path="output_dir"

PATH_TO_IMAGES_DIR = pathlib.Path(image_dir)
IMAGE_PATHS = sorted(list(PATH_TO_IMAGES_DIR.glob(image_file_pattern)))

# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
class_num =len(category_index)

def get_color_table(class_num, seed=0):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

colortable = get_color_table(class_num)

# # Model preparation and loading the model from the disk
def load_model(model_path):

    model_dir = pathlib.Path(model_path) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

# Predict objects and bounding boxes and format the result
def run_inference_for_single_image(model, image):

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run prediction from the model
    output_dict = model(input_tensor)

    # Input to model is a tensor, so the output is also a tensor
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def infer_object(model, image_path):
    # Read the image using openCV and create an image numpy
    # The final output image with boxes and labels on it.
    imagename = os.path.basename(image_path)

    image_np = cv2.imread(os.path.abspath(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    # Visualization of the results of a detection.
    for i in range(output_dict['detection_classes'].size):

        box = output_dict['detection_boxes'][i]
        classes = output_dict['detection_classes'][i]
        scores = output_dict['detection_scores'][i]

        if scores > 0.5:
            h = image_np.shape[0]
            w = image_np.shape[1]
            classname = category_index[classes]['name']
            classid =category_index[classes]['id']
            #Draw bounding boxes
            cv2.rectangle(image_np, (int(box[1] * w), int(box[0] * h)), (int(box[3] * w), int(box[2] * h)), colortable[classid], 2)

            #Write the class name on top of the bounding box
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            size = cv2.getTextSize(str(classname) + ":" + str(scores), font, 0.75, 1)[0][0]

            cv2.rectangle(image_np,(int(box[1] * w), int(box[0] * h-20)), ((int(box[1] * w)+size+5), int(box[0] * h)), colortable[classid],-1)
            cv2.putText(image_np, str(classname) + ":" + str(scores),
                    (int(box[1] * w), int(box[0] * h)-5), font, 0.75, (0,0,0), 1, 1)
        else:
            break
    # Save the result image with bounding boxes and class labels in file system
    cv2.imwrite(output_path+"/"+imagename, image_np)
    # cv2.imshow(imagename, image_np)

# Obtain the model object
detection_model = load_model(model_path)

# For each image, call the prediction
for image_path in IMAGE_PATHS:
    infer_object(detection_model, image_path)