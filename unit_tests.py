############################################################
#                      Aaron Lafitte                       #
#                        A01852530                         #
#                        PROJECT 2                         #
#
#
#
# A lot of code was adapted fromt he following sources:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# https://github.com/datitran/object_detector_app/
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10


import tensorflow as tf
import numpy as np
import cv2
import os
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

CURRENT_DIRECTORY_PATH = os.getcwd()
PATH_TO_GRAPH = os.path.join(CURRENT_DIRECTORY_PATH, 'graph', 'my_detection_graph.pb')
PATH_TO_LABELS = os.path.join(CURRENT_DIRECTORY_PATH, 'data', 'label_map.pbtxt')

PATH_TO_NINE = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'nine_test.JPG')
PATH_TO_TEN = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'ten_test.JPG')
PATH_TO_ACE = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'ace_test.JPG')
PATH_TO_JACK = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'jack_test.JPG')
PATH_TO_QUEEN = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'queen_test.JPG')
PATH_TO_KING = os.path.join(CURRENT_DIRECTORY_PATH, 'unit_test_images', 'king_test.JPG')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=6, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def detectImage(img_path, name):
    image = cv2.imread(img_path)
    frame_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.70)

    cv2.imshow(name + ' - Aaron Lafitte', image)


detectImage(PATH_TO_NINE, 'Nine Test')
detectImage(PATH_TO_TEN, 'Ten Test')
detectImage(PATH_TO_ACE, 'Ace Test')
detectImage(PATH_TO_JACK, 'Jack Test')
detectImage(PATH_TO_QUEEN, 'Queen Test')
detectImage(PATH_TO_KING, 'King Test')

cv2.waitKey(0)
cv2.destroyAllWindows()
    
