import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QMovie, QIcon, QColor

class VideoThread(QThread):
    object_detected = pyqtSignal(list)

    def __init__(self):
        super().__init__()

    def run(self):
        tres = 0.5
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        classNames = []
        with open('CocoModel/coco.names') as f:
            classNames = f.read().rstrip('\n').split('\n')
        
        configpath = 'CocoModel/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightpath = 'CocoModel/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightpath, configpath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0/127.5)
        net.setInputMean(127.5)
        net.setInputSwapRB(True)

        while True:
            ret, frame = cap.read()
            classIds, conf, bbox = net.detect(frame, confThreshold=tres)

            detected_classes = set()
            
            if len(classIds) != 0:
                for Id, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
                    class_name = classNames[Id - 1]
                    detected_classes.add(class_name)
            else:
                detected_classes.add("No Objects Detected")

            detected_classes_list = list(detected_classes)
            self.object_detected.emit(detected_classes_list)
            
            # cv2.imshow('Screen', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.video_thread = VideoThread()

        self.setGeometry(100, 100, 1920, 1080)
        self.setWindowTitle("CyICMS")
        self.setStyleSheet("background-color: #000000;")

        self.logo_path = "Images/Logo.png"
        original_pixmap = QPixmap(self.logo_path)
        target_size = original_pixmap.size().scaled(300, 300, Qt.KeepAspectRatio)
        resized_pixmap = original_pixmap.scaled(target_size, Qt.KeepAspectRatio)
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(resized_pixmap)
        self.logo_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        gif_path = "Images/Face.gif"
        gif_label = QLabel(self)
        gif_movie = QMovie(gif_path)
        gif_label.setMovie(gif_movie)
        gif_movie.start()

        gif_label.setFixedSize(240, 240)
        gif_label.setScaledContents(True)

        left_coloumn = QWidget(self)
        left_coloumn.setStyleSheet("background-color: #171B2E;")

        left_layout = QVBoxLayout(left_coloumn)
        left_layout.addWidget(self.logo_label, alignment=Qt.AlignTop)

        spacer_item_top = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer_item_top)

        left_layout.addWidget(gif_label, alignment=Qt.AlignVCenter | Qt.AlignHCenter)

        spacer_item_between = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)
        left_layout.addItem(spacer_item_between)

        self.start_monitoring_button = QPushButton("Start Monitoring", self)
        self.start_monitoring_button.setStyleSheet("background-color: #3498db; color: #ffffff; font-weight: bold;")
        self.start_monitoring_button.setFixedSize(260, 40)
        left_layout.addWidget(self.start_monitoring_button, alignment=Qt.AlignVCenter | Qt.AlignHCenter)

        spacer_item_bottom = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer_item_bottom)

        video_button = QPushButton(self)
        video_button.setIcon(QIcon("Icons/video.png"))
        video_button.setFixedSize(40, 40)

        speaker_button = QPushButton(self)
        speaker_button.setIcon(QIcon("Icons/speaker.png"))
        speaker_button.setFixedSize(40, 40)

        bottom_layout = QHBoxLayout()

        bottom_layout.addWidget(speaker_button, alignment=Qt.AlignBottom | Qt.AlignLeft)
        bottom_layout.addSpacing(200)
        bottom_layout.addWidget(video_button, alignment=Qt.AlignBottom | Qt.AlignRight)

        left_layout.addLayout(bottom_layout)

        right_layout = QVBoxLayout()

        text_label = QLabel("INTELLIGENT CABIN MANAGEMENT SYSTEM", self)
        text_label.setStyleSheet("color: #ffffff; font-size: 32px; font-weight: bold;")
        right_layout.addWidget(text_label, alignment=Qt.AlignTop | Qt.AlignRight)

        spacer_item = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout.addItem(spacer_item)

        behaviour_rect_widget = QWidget(self)
        behaviour_rect_widget.setFixedSize(1480, 920)
        behaviour_rect_widget.setStyleSheet("background-color: #333333; border-radius: 10px;")

        behaviour_label = QLabel("BEHAVIOUR MONITORING", behaviour_rect_widget)
        behaviour_label.setStyleSheet("color: #00C8F0; font-size: 32px; font-weight: bold;")
        behaviour_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        behaviour_layout = QVBoxLayout(behaviour_rect_widget)
        behaviour_layout.addWidget(behaviour_label)
        heading_spacer = QSpacerItem(0, 60, QSizePolicy.Expanding, QSizePolicy.Minimum)
        behaviour_layout.addItem(heading_spacer)

        Empty_pixmap = QPixmap('Icons/Empty.png')
        Empty_pixmap = Empty_pixmap.scaled(120, 120, Qt.KeepAspectRatio)
        
        Aggressive_pixmap = QPixmap('Icons/angry.png')
        Aggressive_pixmap = Aggressive_pixmap.scaled(90, 90, Qt.KeepAspectRatio)
        
        Non_Aggressive_pixmap = QPixmap('Icons/smile.png')
        Non_Aggressive_pixmap = Non_Aggressive_pixmap.scaled(90, 90, Qt.KeepAspectRatio)

        horizontal_layout_A = QHBoxLayout()
        horizontal_layout_A.addSpacing(60)

        rectangle_A2 = QLabel()
        rectangle_A2.setStyleSheet("background-color: #000000; border-radius: 5px;")
        rectangle_A2.setFixedSize(400, 200)
        self.label_A2 = QLabel("SEAT: A2")
        self.label_A2.setStyleSheet("color: #00C8F0; font-size: 18px; font-weight: bold;")
        self.image_label_A2 = QLabel()
        self.image_label_A2.setAlignment(Qt.AlignCenter)
        self.image_label_A2.setPixmap(Empty_pixmap)
        self.status_text_A2 = QLabel("Status: Empty")
        self.status_text_A2.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.status_text_A2.setAlignment(Qt.AlignCenter)
        self.object_text_A2 = QLabel("")
        self.object_text_A2.setStyleSheet("color: Red; font-weight: bold; font-size: 20px;")
        self.object_text_A2.setAlignment(Qt.AlignCenter)

        layout_A2 = QVBoxLayout()
        layout_A2.addWidget(self.label_A2, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout_A2.addWidget(self.image_label_A2, alignment=Qt.AlignCenter)
        layout_A2.addWidget(self.status_text_A2, alignment=Qt.AlignCenter)
        layout_A2.addWidget(self.object_text_A2, alignment=Qt.AlignCenter)
        layout_A2.addStretch(1)
        rectangle_A2.setLayout(layout_A2)

        horizontal_layout_A.addWidget(rectangle_A2)
            
        rectangle_A1 = QLabel()
        rectangle_A1.setStyleSheet("background-color: #000000; border-radius: 5px;")
        rectangle_A1.setFixedSize(400, 200)
        self.label_A1 = QLabel("SEAT: A1")
        self.label_A1.setStyleSheet("color: #00C8F0; font-size: 18px; font-weight: bold;")
        self.image_label_A1 = QLabel()
        self.image_label_A1.setAlignment(Qt.AlignCenter)
        self.image_label_A1.setPixmap(Empty_pixmap)
        self.status_text_A1 = QLabel("Status: Empty")
        self.status_text_A1.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.status_text_A1.setAlignment(Qt.AlignCenter)
        self.object_text_A1 = QLabel("")
        self.object_text_A1.setStyleSheet("color: Red; font-weight: bold; font-size: 20px;")
        self.object_text_A1.setAlignment(Qt.AlignCenter)

        layout_A1 = QVBoxLayout()
        layout_A1.addWidget(self.label_A1, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout_A1.addWidget(self.image_label_A1, alignment=Qt.AlignCenter)
        layout_A1.addWidget(self.status_text_A1, alignment=Qt.AlignCenter)
        layout_A1.addWidget(self.object_text_A1, alignment=Qt.AlignCenter)
        layout_A1.addStretch(1)
        rectangle_A1.setLayout(layout_A1)

        horizontal_layout_A.addWidget(rectangle_A1)

        behaviour_layout.addLayout(horizontal_layout_A)
        layout_spacer = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        behaviour_layout.addItem(layout_spacer)

        horizontal_layout_B = QHBoxLayout()
        horizontal_layout_B.addSpacing(60)

        rectangle_B2 = QLabel()
        rectangle_B2.setStyleSheet("background-color: #000000; border-radius: 5px;")
        rectangle_B2.setFixedSize(400, 200)
        self.label_B2 = QLabel("SEAT: B2")
        self.label_B2.setStyleSheet("color: #00C8F0; font-size: 18px; font-weight: bold;")
        self.image_label_B2 = QLabel()
        self.image_label_B2.setAlignment(Qt.AlignCenter)
        self.image_label_B2.setPixmap(Empty_pixmap)
        self.status_text_B2 = QLabel("Status: Empty")
        self.status_text_B2.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.status_text_B2.setAlignment(Qt.AlignCenter)
        self.object_text_B2 = QLabel("")
        self.object_text_B2.setStyleSheet("color: Red; font-weight: bold; font-size: 20px;")
        self.object_text_B2.setAlignment(Qt.AlignCenter)

        layout_B2 = QVBoxLayout()
        layout_B2.addWidget(self.label_B2, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout_B2.addWidget(self.image_label_B2, alignment=Qt.AlignCenter)
        layout_B2.addWidget(self.status_text_B2, alignment=Qt.AlignCenter)
        layout_B2.addWidget(self.object_text_B2, alignment=Qt.AlignCenter)
        layout_B2.addStretch(1)
        rectangle_B2.setLayout(layout_B2)

        horizontal_layout_B.addWidget(rectangle_B2)

        rectangle_B1 = QLabel()
        rectangle_B1.setStyleSheet("background-color: #000000; border-radius: 5px;")
        rectangle_B1.setFixedSize(400, 200)
        self.label_B1 = QLabel("SEAT: B1")
        self.label_B1.setStyleSheet("color: #00C8F0; font-size: 18px; font-weight: bold;")
        self.image_label_B1 = QLabel()
        self.image_label_B1.setAlignment(Qt.AlignCenter)
        self.image_label_B1.setPixmap(Empty_pixmap)
        self.status_text_B1 = QLabel("Status: Empty")
        self.status_text_B1.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.status_text_B1.setAlignment(Qt.AlignCenter)
        self.object_text_B1 = QLabel("")
        self.object_text_B1.setStyleSheet("color: Red; font-weight: bold; font-size: 20px;")
        self.object_text_B1.setAlignment(Qt.AlignCenter)

        layout_B1 = QVBoxLayout()
        layout_B1.addWidget(self.label_B1, alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout_B1.addWidget(self.image_label_B1, alignment=Qt.AlignCenter)
        layout_B1.addWidget(self.status_text_B1, alignment=Qt.AlignCenter)
        layout_B1.addWidget(self.object_text_B1, alignment=Qt.AlignCenter)
        layout_B1.addStretch(1)
        rectangle_B1.setLayout(layout_B1)

        horizontal_layout_B.addWidget(rectangle_B1)

        behaviour_layout.addLayout(horizontal_layout_B)
        
        behaviour_layout.addSpacing(60)

        self.msg_rect = QLabel()
        self.msg_rect.setStyleSheet("background-color: #000000; border-radius: 5px;")
        self.msg_rect.setFixedSize(680, 220)
        self.heading = QLabel("OBJECT MONITORING")
        self.heading.setStyleSheet("color: #00C8F0; font-size: 26px; font-weight: bold;")

        self.object_msg = QLabel("No Objects Detected")
        self.object_msg.setStyleSheet("color: #ffffff; font-size: 26px; font-weight: bold;")

        msg_layout = QVBoxLayout()
        msg_layout.addWidget(self.heading, alignment=Qt.AlignTop | Qt.AlignHCenter)
        msg_layout.addWidget(self.object_msg, alignment=Qt.AlignTop | Qt.AlignCenter)
        self.msg_rect.setLayout(msg_layout)
        behaviour_layout.addWidget(self.msg_rect, alignment=Qt.AlignHCenter)
        behaviour_layout.addSpacing(40)

        right_layout.addWidget(behaviour_rect_widget, alignment=Qt.AlignTop | Qt.AlignRight)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(left_coloumn, alignment=Qt.AlignLeft)
        main_layout.addWidget(right_widget, alignment=Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(main_layout)

        self.start_monitoring_button.clicked.connect(self.start_monitoring)
        self.video_thread.object_detected.connect(self.update_object_label)

    def update_object_label(self, object_names):
        red_objects = ['scissors', 'knife', 'baseball bat', 'fork']
        if isinstance(object_names, list):
            colors = []
            for obj in object_names:
                if obj in red_objects:
                    colors.append("red")
                else:
                    colors.append("white")
            
            colored_object_names = [f"<font color='{color}'>{obj}</font>" for obj, color in zip(object_names, colors)]
            object_names_str = ", ".join(colored_object_names)
            self.object_msg.setText(object_names_str)
        else:
            if object_names in red_objects:
                self.object_msg.setText(f"<font color='red'>{object_names}</font>")
            else:
                self.object_msg.setText(str(object_names))
            
    def start_monitoring(self):
        self.start_monitoring_button.setEnabled(False)
        self.video_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())