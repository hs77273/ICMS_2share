import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5.QtGui import QPixmap,QImage
import pathlib
import json

class Config:
    def __init__(self):
        current = pathlib.Path(__file__).parent.resolve()

        with open(current.joinpath("config.json")) as data_file:
            data = json.load(data_file)

        self.camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        self.camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
        self.camera_source_3 = data["CAMERA"]["THIRD_CAMERA_INDEX"]
        self.seat_coordinates = seats_coordinates(data["SEAT_COORDINATES"], data["FRAME_SHAPE"])

class objectsFiles:
    def __init__(self):
        classNames = []
        with open('CocoModel/coco.names') as f:
            classNames = f.read().rstrip('\n').split('\n')

        filter = []
        with open('CocoModel/filter.names') as f:
            filter = f.read().rstrip('\n').split('\n')
            
        offensive_objects = []
        with open('CocoModel/offensive.names') as f:
            offensive_objects = f.read().rstrip('\n').split('\n')
            
        yolov5path = 'CocoModel/yolov5s.pt'

        self.classNames = classNames
        self.filter = filter
        self.offensive_objects = offensive_objects
        self.yolov5path = yolov5path
        
class CameraWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, frame=None, parent=None):
        super().__init__(parent)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        if frame is not None:
            self.update_frame(frame)

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.closed.emit()
            
def seats_coordinates(data, frame_shape):
    h, w, d = frame_shape
    return [(int(coord[0] * w), int(coord[1] * h), int(coord[2] * w), int(coord[3] * h), seat_name) for seat_name, coord in data.items()]
            
def draw_seats(frame, seat_coordinates):
    color = (0, 0, 255)

    for x1, y1, x2, y2, name in seat_coordinates:
        cv2.rectangle(frame, (x1, y2), (x2, y1), color, 5)
        cv2.putText(frame, f"Seat {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame
