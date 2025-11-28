import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLineEdit, QComboBox, QDialog, QMainWindow
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from layout_colorwidget import Color


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB defect detection")
        self.setMinimumSize(1000, 600)
        self.image = None
        self.processed_image = None

        self.setStyleSheet("background-color: white;")
        #main layout
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        panel_layout = QHBoxLayout()
        button_layout = QVBoxLayout()
        defects_layout = QVBoxLayout()

        '''
        widget = QWidget()
        widget.setLayout(image_layout)
        self.setCentralWidget(widget)
        '''
        self.original_label = QLabel("PCB Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(480, 480)

        self.processed_label = QLabel("Image with detected defects")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setFixedSize(480, 480)


        image_layout.addWidget(self.original_label)
        main_layout.addLayout(image_layout)
        image_layout.addWidget(self.processed_label)


        btn1_load = QPushButton("Load Image")
        btn1_load.clicked.connect(self.load_image)
        button_layout.addWidget(btn1_load)

        btn2_load = QPushButton("Load Image")
        btn2_load.clicked.connect(self.load_image)
        button_layout.addWidget(btn2_load)

        #Combobox
        '''
        self.pageCombo = QComboBox()
        self.pageCombo.addItems(["Missing Hole", "Mouse bite", "Open circuit", "Short", "Spur", "Spurious copper"])
        defects_layout.addWidget(self.pageCombo)
        '''
        #Legenda z tytułem
        legend_title = QLabel("Legend")
        font = legend_title.font()
        font.setPointSize(15)
        legend_title.setFont(font)
        legend_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        defects_layout.addWidget(legend_title)

        legend = QLabel(" Missing Hole \n Mouse bite \n Open circuit \n Short \n Spur \n Spurious copper")
        font = legend.font()
        font.setPointSize(10)
        legend.setFont(font)
        legend.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        defects_layout.addWidget(legend)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(panel_layout)
        panel_layout.addLayout(button_layout)
        panel_layout.addLayout(defects_layout)
        self.setLayout(main_layout)


    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.show_image(self.image, self.original_label)

    def show_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)


# do bondingbox
    def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

  


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())