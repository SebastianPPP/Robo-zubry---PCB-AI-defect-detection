import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QComboBox, QDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB defect detection")
        self.setMinimumSize(1000, 600)
        self.image = None
        self.processed_image = None

        main_layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        self.original_label = QLabel("PCB Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(480, 480)
        image_layout.addWidget(self.original_label)
        main_layout.addLayout(image_layout)


        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        main_layout.addWidget(btn_load)

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

  




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())