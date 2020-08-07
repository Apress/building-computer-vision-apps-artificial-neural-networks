
# example of loading the keras facenet model
import tensorflow.compat.v1 as tf
from mtcnn import MTCNN

tf.disable_v2_behavior()
import cv2



# load the model
model = tf.keras.models.load_model('/Users/sansari/Downloads/keras-facenet/model/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)

image = cv2.imread("/User/sansari/Downloads/test_images/Egyptian_Mau_219.jpg")
detector = MTCNN()
# detect faces in the image
results = detector.detect_faces(image)

# extract the bounding box from the first face
x1, y1, width, height = results[0]['box']
# bug fix
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height
print(x1, x2, y1, y2)
