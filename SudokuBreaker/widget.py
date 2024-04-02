# This Python file uses the following encoding: utf-8
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtUiTools import loadUiType
generated_class, base_class = loadUiType("dialog.ui")

from PySide6.QtMultimedia import (QAudioInput, QCamera, QCameraDevice,
                                  QImageCapture, QMediaCaptureSession,
                                  QMediaDevices, QMediaMetaData,
                                  QMediaRecorder)
from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox
from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
from PySide6.QtCore import QDateTime, QDir, QTimer, Qt, Slot, qWarning, QIODevice

from metadatadialog import MetaDataDialog
from imagesettings import ImageSettings
from videosettings import VideoSettings, is_android

from ui_camera import Ui_Camera

class MainDialog(QDialog):
    def __init__(self):
        super().__init__()
        widget = base_class()
        self.ui = generated_class()
        self.ui.setupUi(self)

        # Initialize the camera
        self.camera = QCamera()

        # Connect signals
        self.image_capture.imageCaptured.connect(self.display_image)

        # Start the camera
        self.camera.start()

    def display_image(self, id, preview):
        image = preview.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(image)

    def closeEvent(self, event):
        # Stop the camera when the application exits
        self.camera.stop()
        event.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainDialog()
    window.show()
    sys.exit(app.exec())
