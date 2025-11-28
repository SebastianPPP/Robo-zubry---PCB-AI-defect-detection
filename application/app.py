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
        self.setWindowTitle("Zadania z przetwarzania obrazu")
        self.setMinimumSize(1000, 600)
        self.image = None
        self.processed_image = None

        main_layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        self.original_label = QLabel("Oryginalny obraz")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(480, 480)
        image_layout.addWidget(self.original_label)
        main_layout.addLayout(image_layout)


        btn_load = QPushButton("Wczytaj obraz")
        btn_load.clicked.connect(self.load_image)
        main_layout.addWidget(btn_load)


 







        self.setLayout(main_layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Otwórz obraz", "", "Images (*.png *.jpg *.bmp)")
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

    def show_temp_window(self, image, title="Podgląd"):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(400, 400)
        label = QLabel(dialog)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        h, w = image.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(380, 380, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()


        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        method = self.edge_combo.currentText()

        if method == "Roberts":
            kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            edge = cv2.magnitude(grad_x, grad_y)
        elif method == "Prewitt":
            kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            grad_x = cv2.filter2D(gray, -1, kernel_x)
            grad_y = cv2.filter2D(gray, -1, kernel_y)
            edge = cv2.magnitude(np.float32(grad_x), np.float32(grad_y))
        elif method == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge = cv2.magnitude(sobelx, sobely)
        elif method == "LoG":
            blur = cv2.GaussianBlur(gray, (5, 5), 0)  
            laplacian = cv2.Laplacian(blur, cv2.CV_64F)
            edge = cv2.convertScaleAbs(laplacian, alpha=2.0) 

        edge = cv2.convertScaleAbs(edge)
        result = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        self.processed_image = result
        self.show_image(result, self.processed_label)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())