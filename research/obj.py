import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util
from com_in_utils.utils import encodeImage


class CardsDetector:
    def __init__(self, imagePath, modelPath):
        sys.path.append('..')

        # name of the directory containing the object detection module we're using
        self.MODEL_NAME = modelPath
        self.IMAGE_NAME = imagePath

        # get path to current working directory
        CWD_PATH = os.getcwd()

        # path to frozen detection graph.pb  file which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(
            CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')

        # path to label map file
        self.PATH_TO_LABELS = os.path.join(
            CWD_PATH, 'research/data', 'labelmap.pbtxt')

        # path to images
        self.PATH_TO_IMAGE = os.path.join(
            CWD_PATH, 'research', self.IMAGE_NAME)

        # number of classes the object detector can identify
        self.NUM_CLASSES = 6
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)

        self.category_index = label_map_util.create_category_index(
            self.categories)

        self.class_names_mapping = {
            1: "Nine", 2: "Ten", 3: "jack", 4: "queen", 5: "King", 6: "Ace"}

        # load frozen tf model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # define input and output tensors (data) for the object detection classifier
        # input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')

        # each score represents a level of confidence for each of the objects
        # the score id shown on the result image, together woth the class label
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')

        # number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def getPrediction(self):
        # Load the Tensorflow model into memory.
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3] i.e. a single-column array, where each item in the column has the pixel RGB value
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)

        # perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([self.detection_boxes, self.detection_scores,
                                                  self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > .40:
                res.append(idx)

        top_classes = classes.flatten()

        # selecting class 2 and 3
        res_list = [top_classes[i] for i in res]

        class_final_names = [self.class_names_mapping[x] for x in res_list]
        top_scores = [e for l2 in scores for e in l2 if e > 0.30]

        new_scores = classes.flatten()
        new_boxes = boxes.reshape(300, 4)

        # get the boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh = .30

        listOfOutput = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valDict = {}
            valDict["className"] = name
            valDict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > min_score_thresh:
                val = list(new_boxes[i])
                valDict["yMin"] = str(val[0])
                valDict["xMin"] = str(val[1])
                valDict["yMax"] = str(val[2])
                valDict["xMax"] = str(val[3])
                listOfOutput.append(valDict)

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        output_filename = 'output4.jpg'
        cv2.imwrite(output_filename, image)
        opencodedbase64 = encodeImage("output4.jpg")

        listOfOutput.append({"image": opencodedbase64.decode('utf-8')})
        return listOfOutput
