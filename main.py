import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QLineEdit, QGroupBox, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from utils.data_input import capture_images
from utils.trainer import train_model
from utils.recognizer import recognize_from_image, recognize_from_camera


class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.setGeometry(600, 300, 520, 450)
        self.setStyleSheet("background-color: #f0f4f7;")

        self.init_ui()

    def init_ui(self):
        title = QLabel("人脸识别系统", self)
        title.setFont(QFont("微软雅黑", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("请输入姓名用于采集")
        self.name_input.setFont(QFont("微软雅黑", 11))
        self.name_input.setStyleSheet("padding: 6px; border-radius: 8px; border: 1px solid gray;")

        self.capture_btn = QPushButton("采集人脸数据", self)
        self.capture_btn.clicked.connect(self.capture_data)

        self.train_btn = QPushButton("训练模型", self)
        self.train_btn.clicked.connect(train_model)

        self.img_btn = QPushButton("识别上传图片", self)
        self.img_btn.clicked.connect(self.recognize_image)

        self.cam_btn = QPushButton("实时摄像头识别", self)
        self.cam_btn.clicked.connect(recognize_from_camera)

        # 按钮样式统一美化
        for btn in [self.capture_btn, self.train_btn, self.img_btn, self.cam_btn]:
            btn.setFont(QFont("微软雅黑", 11))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    padding: 10px;
                    border: none;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #357ab8;
                }
            """)

        # 组合框体
        form_group = QGroupBox(self)
        form_layout = QVBoxLayout()
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(self.capture_btn)
        form_layout.addWidget(self.train_btn)
        form_layout.addWidget(self.img_btn)
        form_layout.addWidget(self.cam_btn)
        form_group.setLayout(form_layout)
        form_group.setStyleSheet("""
            QGroupBox {
                font: bold 12pt "微软雅黑";
                border: 1px solid gray;
                border-radius: 9px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        main_layout.addSpacing(10)
        main_layout.addWidget(form_group)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

    def capture_data(self):
        name = self.name_input.text().strip()
        if name:
            QMessageBox.information(self, "提示", f"已开始采集 {name} 的人脸数据")
            capture_images(name)
        else:
            QMessageBox.warning(self, "警告", "请输入姓名")

    def recognize_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            recognize_from_image(file_name)


if __name__ == "__main__":
    print("欢迎使用人脸识别系统")
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())
