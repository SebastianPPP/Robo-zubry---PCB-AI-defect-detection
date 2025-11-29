import sys
import cv2
from prediction import predict_img
from const import YOLO_11N_320_E10, YOLO_V8N_160_E8, YOLO_V11_SMALL, BINARY, GRAYSCALE, YOLO_V5NU
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import torch


#from tensorflow.keras.models import load_model


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB DEFECT DETECTION")
        self.setMinimumSize(1000, 600)
        self.image = None
        self.processed_image = None
        self.examples=0
        self.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgb(61, 217, 245), stop:1 rgb(240, 53, 218));")
        
        #main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.title_layout = QHBoxLayout()
        self.image_layout = QHBoxLayout()
        self.panel_layout = QHBoxLayout()
        self.button_layout = QVBoxLayout()
        self.defects_layout = QVBoxLayout()
        

        self.image_title = QLabel("PCB Defects Detection")
        self.image_title.setFixedHeight(40)
        self.image_title.setStyleSheet("""
                color: #000000;               
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 25px;
                """)
        font = self.image_title.font()
        font.setPointSize(15)
        self.image_title.setFont(font)
        self.image_title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.title_layout.addWidget(self.image_title)
        self.main_layout.addLayout(self.title_layout)



        self.original_label = QLabel("Original PCB Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        #self.original_label.setFixedHeight(400)
        #self.original_label.setFixedSize(400, 400)
        self.original_label.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 18px;
                """)

        self.processed_label = QLabel("Image with Detected Defects")
        self.processed_label.setAlignment(Qt.AlignCenter)
        #self.processed_label.setFixedHeight(400)
        #self.processed_label.setFixedSize(400, 400)
        self.processed_label.setStyleSheet("""
            background-color: #262626;
                color: #FFFFFF;
                border-radius: 12px;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 18px;
                """)

        self.image_layout.addWidget(self.original_label)
        self.main_layout.addLayout(self.image_layout)
        self.image_layout.addWidget(self.processed_label)


        combobox_title = QLabel("Choose Model")
        combobox_title.setFixedHeight(50)
        combobox_title.setStyleSheet("""
            background-color: #262626;
            qproperty-alignment: AlignCenter;
            border-radius: 12px;
            padding 20 px;
            color: #FFFFFF;
            font: 75 10pt "Microsoft YaHei UI";
            font-weight: bold;
            font-size: 18px;
            """)
        #font = combobox_title.font()
        #font.setPointSize(15)
        #combobox_title.setFont(font)
        #combobox_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.button_layout.addWidget(combobox_title)

        combobox1 = QComboBox()
        combobox1.addItem('YOLO_11N_320_E10')
        combobox1.addItem('YOLO_V11_SMALL')
        combobox1.addItem('YOLO_V8N_160_E8')
        combobox1.addItem('BINARY')
        combobox1.addItem('GRAYSCALE')
        combobox1.addItem('YOLO_V5NU')
        combobox1.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 18px;
                """)
        self.button_layout.addWidget(combobox1)
        combobox1.activated.connect(self.activated)

        btn1_load = QPushButton("Load Image")
        btn1_load.clicked.connect(self.load_image)
        btn1_load.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                padding: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 18px;
                """)
        self.button_layout.addWidget(btn1_load)


        btn2_load = QPushButton("Examples")
        btn2_load.clicked.connect(self.load_example_image)
        btn2_load.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                padding: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                font-size: 18px;
                """)
        self.button_layout.addWidget(btn2_load)
        
        #Legenda z tytułem
        legend_title = QLabel("Legend")
        legend_title.setFixedHeight(40)
        legend_title.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
            font-size: 18px;
            """)
        font = legend_title.font()
        font.setPointSize(15)
        legend_title.setFont(font)
        legend_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.defects_layout.addWidget(legend_title)

        legend = QLabel("🟢 - Mouse bite \n 🟠 - Spur \n 🔴 - Missing Hole \n 🔵 - Short \n 🟡 - Open circuit \n 🟣 - Spurious copper")
        legend.setFixedHeight(150)
        legend.setStyleSheet("""
            background-color: #262626;
                border-radius: 12px;
                color: #FFFFFF;
                font: 75 10pt "Microsoft YaHei UI";
                font-weight: bold;
                """)
        font = legend.font()
        font.setPointSize(10)
        legend.setFont(font)
        legend.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.defects_layout.addWidget(legend)

        
        
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.panel_layout)
        self.panel_layout.addLayout(self.button_layout)
        self.panel_layout.addLayout(self.defects_layout)
        self.setLayout(self.main_layout)

    def activated(self, index):
        global model
        if (index == 0):
            model = YOLO_11N_320_E10
            #loaded_model = load_model("model_11n_15e.h5")
            #predict_img(YOLO_11N_320_E10, random_val_img=True)
            print("Activated index:", index)
        if (index == 1):
            model = YOLO_V11_SMALL
            print("Activated index:", index)
        if (index == 2):
            model = YOLO_V8N_160_E8
            print("Activated index:", index)
        if (index == 3):
            model = BINARY
            print("Activated index:", index)
        if (index == 4):
            model = GRAYSCALE
            print("Activated index:", index)
        if (index == 5):
            model = YOLO_V5NU
            print("Activated index:", index)


    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")

        if file_name:
            #boxes, plot, score, label = predict_img(model, file_name)
            #boxes = boxes()
            results = predict_img(model, file_name)
            boxes = results["xyxy"]
            score = results["confs"]
            label = results["classes"]
            self.image = cv2.imread(file_name)
            self.show_image(self.image, self.original_label)
            self.boundingboxes(self.image, boxes, score, label)
            self.show_image(self.image, self.processed_label)


    def load_example_image(self):
        file1 = "data/examples/1.jpg"
        file2 = "data/examples/2.jpg"
        file3 = "data/examples/3.jpg"
        file4 = "data/examples/4.jpg"
        file5 = "data/examples/5.jpg"
        file6 = "data/examples/6.jpg"
        file7 = "data/examples/7.jpg"

        files = [file1, file2, file3, file4, file5, file6, file7]
        file_name = files[self.examples]

        if file_name:
            #boxes, plot, score, label = predict_img(model, file_name)
            #boxes = boxes()
            results = predict_img(model, file_name)
            boxes = results["xyxy"]
            score = results["confs"]
            label = results["classes"]
            self.image = cv2.imread(file_name)
            self.show_image(self.image, self.original_label)
            self.boundingboxes(self.image, boxes, score, label)
            self.show_image(self.image, self.processed_label)
            self.examples +=1
        if (self.examples == 7): self.examples = 0


    def show_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def boundingboxes(self, image, boxes, score, label):
        colors = [(0, 255, 0),(0, 128, 255),(0, 0, 255),(255, 0, 0),(0, 255, 255),(255, 0, 127)]
        labels_dict = {0:"MB", 1:"Sp", 2:"MH", 3:"Sh", 4:"OC", 5:"SC"}

        if not boxes:
            print("Boks stop")
        else:
            #print(score)
            #score = score[1].tolist()
            print(score)

            #label = label[1].tolist()
            print(label)
            i = 0

            for box in boxes:
                box = box.tolist()
                print(box)
                print(type(box)) 
                startpoint = (int(box[0][0]), int(box[0][1]))
                endpoint = (int(box[0][2]), int(box[0][3]))
                score1 = score[i].tolist()
                label1 = label[i].tolist()
                label_img = int(label1[0])
                color = colors[label_img]
                print(color)
                score1 = str(score1[0])
                label_text = labels_dict[label_img]
                label_text = label_text + ", " + score1[:4]
                cv2.rectangle(self.image, startpoint, endpoint, color, thickness=3)
                cv2.putText(self.image, label_text, (int(box[0][0]), int(box[0][1])-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), 6, cv2.LINE_AA)
                cv2.putText(self.image, label_text, (int(box[0][0]), int(box[0][1])-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)
                i+=1
        
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())