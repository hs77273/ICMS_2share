import cv2
from helper import *
import numpy as np
from keras.models import load_model

OBJECT = objectsFiles()
CONFIG = Config()
behaviour_model = load_model("BehaviourModel/behaviour.h5", compile=False)
behaviour_class = ['Aggresive', 'Non-Aggresive', 'NoFace']

class YoloObjectdetection:
    def __init__(self):
        self.tres = 0.5
        self.net = cv2.dnn_DetectionModel(OBJECT.weightpath, OBJECT.configpath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean(127.5)
        self.net.setInputSwapRB(True)
    
    def process_objects(self, frame):
        classIds, conf, bbox = self.net.detect(frame, confThreshold=self.tres)
        detected_classes = set()

        if len(classIds) != 0:
            for Id, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
                class_name = OBJECT.classNames[Id - 1]
                if class_name not in OBJECT.filter:
                    detected_classes.add(class_name)
        else:
            detected_classes.add("No Objects Detected")

        detected_classes_list = list(detected_classes)
        return detected_classes_list
    
class BehaviourDetection:
    def __init__(self,frame):
        self.frame = frame
        self.seat_coordinate = CONFIG.seat_coordinates
        
    def behaviour_process(self,frame):
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = behaviour_model.predict(image, verbose=0)
        index = np.argmax(prediction)
        label_name = behaviour_class[index]
        confidence_score = prediction[0][index]
        return label_name, confidence_score

    def process_behaviour(self):
        behaviour_status = []
        seat_no = []
        for x1, y1, x2, y2, seat_name in self.seat_coordinate:
            try:
                seat = self.frame[y2:y1, x1:x2]
                class_name, confidence_score = self.behaviour_process(seat)
                label = f" {seat_name} ::  {class_name}: {round(confidence_score * 100, 2)}%"
                behaviour_status.append(class_name)
                seat_no.append(seat_name)
            except Exception as e:
                print(e)
        
        return seat_no,behaviour_status