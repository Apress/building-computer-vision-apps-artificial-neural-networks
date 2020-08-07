import os
import pathlib
import random
import numpy as np
import tensorflow as tf
import cv2
import threading

# Import the object detection module.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

from videoasync import VideoCaptureAsync
import tracker as hasher

lock = threading.Lock()

# to make gfile compatible with v2
tf.gfile = tf.io.gfile

model_path = "./../model/ssd_inception_v2_coco_2018_01_28"
labels_path = "./../model/mscoco_label_map.pbtxt"

# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
class_num =len(category_index)+100
object_ids = {}
hasher_object = hasher.ObjectHasher()

#Function to create color table for each object class
def get_color_table(class_num, seed=50):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

colortable = get_color_table(class_num)

#  Initialize and start the asynchronous video capture thread
cap = VideoCaptureAsync().start()

# # Model preparation
def load_model(model_path):
    model_dir = pathlib.Path(model_path) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

model = load_model(model_path)

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

    return output_dict

# Function to draw bounding boxes and tracking information on the image frame
def track_object(model, image_np):
    global object_ids, lock
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
            hash, object_ids = hasher_object.getObjectId(image_np, int(box[1] * w), int(box[0] * h), int(box[3] * w),
                                             int(box[2] * h), object_ids)

            size = cv2.getTextSize(str(classname) + ":" + str(scores)+"[Id: "+str(object_ids.get(hash))+"]", font, 0.75, 1)[0][0]

            cv2.rectangle(image_np,(int(box[1] * w), int(box[0] * h-20)), ((int(box[1] * w)+size+5), int(box[0] * h)), colortable[classid],-1)
            cv2.putText(image_np, str(classname) + ":" + str(scores)+"[Id: "+str(object_ids.get(hash))+"]",
                    (int(box[1] * w), int(box[0] * h)-5), font, 0.75, (0,0,0), 1, 1)

            cv2.putText(image_np, "Number of objects detected: "+str(len(object_ids)),
                        (10,20), font, 0.75, (0, 0, 0), 1, 1)
        else:
            break
    return image_np

# Function to implement infinite while loop to read video frames and generate the output for web browser
def streamVideo():
    global lock
    while (True):
        retrieved, frame = cap.read()
        if retrieved:
            with lock:
                frame = track_object(model, frame)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.stop()
            cv2.destroyAllWindows()
            break

    # When everything done, release the capture
    cap.stop()
    cv2.destroyAllWindows()
