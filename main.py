from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from datetime import datetime
from ControlClient import ClientRequests

from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

import sys
import time
import os
import torch
import threading
import socket
import numpy as np
import ctypes

myappid = u'skyscopeID' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        uic.loadUi("MainWindow.ui", self) #load ui file
        self.showMaximized()

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) # Remove default title bar
        self.titleFrame.mouseMoveEvent = self.MoveWindow  # Set custom navigation bar
        self.closeButton.clicked.connect(lambda: MainApp.exit())     # Assign exit button
        self.minButton.clicked.connect(lambda: self.showMinimized()) # Assign minimized button
        self.maxButton.clicked.connect(lambda: self.showMaximized()) # Assign maximized button      
        self.importInfoButton.clicked.connect(self.importInfoButtonEvent) # Assign info button in Import tab
        self.streamInfoButton.clicked.connect(self.streamInfoButtonEvent) # Assign info button in Stream tab

        self.widget1.currentChanged.connect(self.handleTabChange)

        self.baseDir = os.path.dirname(os.path.abspath(sys.executable))
        self.project_path = os.path.join(self.baseDir, "SkyScope Projects")

        self.date_string = datetime.now().strftime("%Y-%m-%d")

        self.inspectionDetailsFlag = False  #initialize inspection details not filled
        self.videoAvailable = False #initial state - no video imported
        self.imageAvailable = False #initial state - no image imported
        self.modelLoaded = False #initial state - no model loaded

        self.directoryButton.clicked.connect(self.setDirectoryEvent)

        # UI initializations
        self.scrubberBar.setVisible(False)
        self.videoWidgets.setVisible(False)
        self.dateText.setText(f"{self.date_string}")
        self.directoryText.setText(f"{self.project_path}")
        self.streamDirectory.setText(f"{self.project_path}")
        self.modelStatus.setVisible(False)

        # Get text from text input fields
        self.location_text_edit = self.findChild(QtWidgets.QTextEdit, "locationText")
        self.inspector_text_edit = self.findChild(QtWidgets.QTextEdit, "inspectorText")
        self.serialnum_text_edit = self.findChild(QtWidgets.QTextEdit, "serialnumText")
        self.height_text_edit = self.findChild(QtWidgets.QTextEdit, "heightText")
        self.voltage_text_edit = self.findChild(QtWidgets.QTextEdit, "voltageText")

        # Connect import tab button events
        self.saveDetails.clicked.connect(self.saveInformationEvent) 
        self.importButton.clicked.connect(self.importButtonEvent) 
        self.skipButton.clicked.connect(self.skipFootageEvent)  
        self.backButton.clicked.connect(self.backFootageEvent) 
        self.playVidButton.clicked.connect(self.playPauseVideoEvent)
        self.captureButton.clicked.connect(self.captureButtonEvent)
        self.runButton.clicked.connect(self.runButtonEvent)
        self.newFile.clicked.connect(self.newFileButtonEvent)
        self.modelButton.clicked.connect(self.load_model) # Import model detect.pt (CPU Only)

        # Initialize the video player variables
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.timer = None
        self.paused = False
        self.current_frame = 0
        self.newFile.setEnabled(False)
        self.fileType.currentIndexChanged.connect(self.fileTypeChanged)

        # Stream tab initializations
        self.camera = None
        self.streamThread = None
        self.ipAddrText.setVisible(False) #remove IP Addr text box by default
        self.is_recording = False # initialize recording state
        self.ip_text_edit = self.findChild(QtWidgets.QTextEdit, "ipAddrText")
        self.defIP = '192.168.43.1'
        self.hasIP = False
        self.client = None
        self.isStreaming = False
        self.defDeviceCam = 0  #web camera

        # Connect stream tab button events
        self.recordStream.clicked.connect(self.recordStreamEvent)
        self.captureStream.clicked.connect(self.captureStreamEvent)
        self.stopStream.clicked.connect(self.stopStreamEvent)
        self.startStream.clicked.connect(self.startStreamEvent)

        # Search for available cameras
        self.available_cameras = QCameraInfo.availableCameras() 
        self.available_cameras.pop() #if no camera found
        if not self.available_cameras: sys.exit()

        self.cameraDropdown.addItems([camera.description() for camera in self.available_cameras])

        # Recording update elapsed time
        self.recordTime.setVisible(False)
        self.timer = QTimer() #timer for video frames update
        self.timer1 = QTimer() #recording elapsed time
        
        # Run Prediction Variables
        self.predictionThread = None
        self.predictionThreadFinished = True

        # YOLOv5 variables
        self.imgsz=(416, 416)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device = select_device('cpu')  # cuda cpu device
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.line_thickness=1  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.model=None  # model file
        self.data = os.path.join(self.baseDir, 'data/coco128.yaml') # dataset.yaml path


    def setDirectoryEvent(self):
        try: 
            directory = QFileDialog(self)
            directory.setFileMode(QFileDialog.Directory)
            directory.setOption(QFileDialog.ShowDirsOnly, True)
        
                
            if directory.exec_() == QFileDialog.Accepted:
                directory_path = directory.selectedFiles()[0]
                if os.path.basename(directory_path) != "SkyScope Projects":
                    if not os.path.exists(os.path.join((directory_path),"SkyScope Projects")):
                        QMessageBox.information(self, "SkyScope Projects folder not found", "A project folder will be created in this directory")
                        os.mkdir(os.path.join((directory_path),"SkyScope Projects"))
                    directory_path = os.path.join((directory_path),"SkyScope Projects")
                    if not os.path.exists(os.path.join((directory_path),"Recordings and Captures")):
                        os.mkdir(os.path.join((directory_path),"Recordings and Captures"))

                self.directoryText.setText(f"{directory_path}")
                self.streamDirectory.setText(f"{directory_path}")

        except Exception as e:
            QMessageBox.warning(self, "Invalid project folder", "No such local directory")
            return
        

    def importInfoButtonEvent(self):
        image_path = os.path.join(self.baseDir, "img\importTabInfo.png")
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is not None:
            # Get image dimensions
            h, w, c = image.shape
        
            # Calculate bytes per line
            bytes_per_line = 3 * w
        
            # Create a QImage from the OpenCV image data
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
            # Create the QLabel to display the image
            self.label = QLabel()
            self.frameDialog = QDialog(self)
            self.frameDialog.setWindowTitle("Import Footage Tab Info")
            self.frameDialog.setFixedSize(1300, 600)
            self.frameDialog.setWindowFlags(self.frameDialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            self.frameDialog.setStyleSheet("background-color: transparent")
        
            # Set up the layout for the dialog
            layout = QVBoxLayout()
            layout.addWidget(self.label)
            self.frameDialog.setLayout(layout)
        
            # Resize the QImage to fit the dialog label
            scaled_qimage = qimage.scaled(self.frameDialog.width(), self.frameDialog.height(), Qt.KeepAspectRatio)
        
            # Set the QPixmap on the QLabel
            self.label.setPixmap(QPixmap.fromImage(scaled_qimage))
            layout.setAlignment(Qt.AlignCenter)
        
            # Show the dialog
            self.frameDialog.exec_()
        else:
            QMessageBox.warning(self, "Information image is missing or tampered", "Failed to load the information image")
            # print("Failed to load the image:", image_path)
        

    def streamInfoButtonEvent(self):
        image_path = os.path.join(self.baseDir, "img\importTabInfo.png")
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        
        if image is not None:
            # Get image dimensions
            h, w, c = image.shape
        
            # Calculate bytes per line
            bytes_per_line = 3 * w
        
            # Create a QImage from the OpenCV image data
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
            # Create the QLabel to display the image
            self.label = QLabel()
            self.frameDialog = QDialog(self)
            self.frameDialog.setWindowTitle("Start Live View Tab Info")
            self.frameDialog.setFixedSize(1300, 600)
            self.frameDialog.setWindowFlags(self.frameDialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            self.frameDialog.setStyleSheet("background-color: transparent")
        
            # Set up the layout for the dialog
            layout = QVBoxLayout()
            layout.addWidget(self.label)
            self.frameDialog.setLayout(layout)
        
            # Resize the QImage to fit the dialog label
            scaled_qimage = qimage.scaled(self.frameDialog.width(), self.frameDialog.height(), Qt.KeepAspectRatio)
        
            # Set the QPixmap on the QLabel
            self.label.setPixmap(QPixmap.fromImage(scaled_qimage))
            layout.setAlignment(Qt.AlignCenter)
        
            # Show the dialog
            self.frameDialog.exec_()
        else:
            QMessageBox.warning(self, "Information image is missing or tampered", "Failed to load the information image")
            # print("Failed to load the image:", image_path)
    

    def saveInformationEvent(self):
        try:
            self.location = self.location_text_edit.toPlainText()
            self.inspector = self.inspector_text_edit.toPlainText()
            self.serialnum = self.serialnum_text_edit.toPlainText()
            self.height = self.height_text_edit.toPlainText()
            self.voltage = self.voltage_text_edit.toPlainText()
            self.directory_text_edit = self.findChild(QtWidgets.QTextEdit, "directoryText")
            self.directory = rf"{self.directory_text_edit.toPlainText()}"

            if self.location == "" or self.serialnum == "": raise Exception("Location or Serial Number is empty")
            if self.inspector == "": self.inspector = "UnnamedInspector"
            self.textFilename = f"{self.serialnum}_{self.inspector}_info_{self.date_string}.txt" # Create a filename for the text
            
            if not os.path.exists(os.path.join((self.directory),f"{self.location}_{self.serialnum}")):
                os.mkdir(os.path.join((self.directory),f"{self.location}_{self.serialnum}"))
            self.directory = os.path.join((self.directory),f"{self.location}_{self.serialnum}")

            if not os.path.exists(os.path.join((self.directory),f"{self.inspector}_{self.date_string}")):
                os.mkdir(os.path.join((self.directory),f"{self.inspector}_{self.date_string}"))
            self.directory = os.path.join((self.directory),f"{self.inspector}_{self.date_string}")

    
            with open(f"{self.directory}\{self.textFilename}", "w") as f:
                f.write(f"Location: {self.location}\n")
                f.write(f"Inspector: {self.inspector}\n")
                f.write(f"Serial Number: {self.serialnum}\n")
                f.write(f"Rated Voltage: {self.voltage}\n")
                f.write(f"Height: {self.height}\n")

                self.inspectionDetailsFlag = True #To allow media import

                # Disable save button
                self.saveDetails.setText(f"Saved")
                self.saveDetails.setEnabled(False)
                self.saveDetails.setStyleSheet(u"QPushButton {\n" "background-color: rgb(39, 39, 54);"
                                                                "color: rgb(255, 255, 255);" "border-radius: 10px;\n}")
                self.saveDetails.setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                
                # Enable new file button
                self.newFile.setEnabled(True)
                self.newFile.setStyleSheet(u"QPushButton {\n" "background-color: rgb(53, 58, 90);"
                                                            "color: rgb(255, 255, 255);" """font: 8pt "Dosis";\n}"""
                                            "QPushButton::Hover {\n" "background-color: rgb(121, 124, 145);\n}")
                self.newFile.setProperty("cursor", QCursor(Qt.ArrowCursor))

                QMessageBox.information(self, f"{self.textFilename} Saved", "Check the current directory for the generated folder and text file")
                
                self.locationText.setPlaceholderText(self.location)
                self.inspectorText.setPlaceholderText(self.inspector)
                self.serialnumText.setPlaceholderText(self.serialnum)
                self.voltageText.setPlaceholderText(self.voltage)
                self.heightText.setPlaceholderText(self.height)
                self.directoryText.setPlaceholderText(self.directory)

                # Set all fields to read-only
                self.location_text_edit.setReadOnly(True)
                self.inspector_text_edit.setReadOnly(True)
                self.serialnum_text_edit.setReadOnly(True)
                self.height_text_edit.setReadOnly(True)
                self.voltage_text_edit.setReadOnly(True)

                # Set the cursor to forbidden
                self.locationText.viewport().setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                self.inspectorText.viewport().setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                self.serialnumText.viewport().setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                self.voltageText.viewport().setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                self.heightText.viewport().setProperty("cursor", QCursor(Qt.ForbiddenCursor))
                
        except Exception as e:
            QMessageBox.warning(self, "Incomplete Information", f"{e}")
            return


    def newFileButtonEvent(self):
        # Disable import tab functions until new inspection details are recorded and saved
        self.inspectionDetailsFlag = False
        if self.paused == False:
            self.playPauseVideoEvent()
        self.imageAvailable = False
        self.videoAvailable = False
        self.scrubberBar.setVisible(False)
        self.videoWidgets.setVisible(False)
        self.mediaLabel.clear()

        # Enable save button
        self.saveDetails.setText(f"Save")
        self.saveDetails.setEnabled(True)
        self.saveDetails.setStyleSheet(u"QPushButton {\n" "background-color: rgb(53, 58, 90);"
                                                        "color: rgb(255, 255, 255);" "border-radius: 10px;\n}"
                                        "QPushButton::Hover {\n" "background-color: rgb(121, 124, 145);\n}")
        self.saveDetails.setProperty("cursor", QCursor(Qt.ArrowCursor))

        # Disable new file button
        self.newFile.setEnabled(False)
        self.newFile.setStyleSheet(u"QPushButton {\n" "background-color: rgb(39, 39, 54);"
                                                    "color: rgb(255, 255, 255);" """font: 8pt "Dosis";\n}""")
        self.newFile.setProperty("cursor", QCursor(Qt.ForbiddenCursor))
        
        #disable editing for the save button
        self.saveDetails.setText(f"Save")
        self.location_text_edit.clear()
        self.locationText.setPlaceholderText("Street, City, Zip")
        self.inspector_text_edit.clear()
        self.inspectorText.setPlaceholderText("Firstname MI Lastname")
        self.serialnum_text_edit.clear()
        self.serialnumText.setPlaceholderText("**********")
        self.voltage_text_edit.clear()
        self.voltageText.setPlaceholderText("")
        self.height_text_edit.clear()
        self.heightText.setPlaceholderText("")
        # Set all fields to read-only
        self.location_text_edit.setReadOnly(False)
        self.inspector_text_edit.setReadOnly(False)
        self.serialnum_text_edit.setReadOnly(False)
        self.height_text_edit.setReadOnly(False)
        self.voltage_text_edit.setReadOnly(False)
        self.locationText.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        self.inspectorText.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        self.serialnumText.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        self.voltageText.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        self.heightText.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))


    def fileTypeChanged(self):
        if self.paused == False:
            self.playPauseVideoEvent()
        self.imageAvailable = False
        self.videoAvailable = False
        self.scrubberBar.setVisible(False)
        self.videoWidgets.setVisible(False)
        self.image_source = None
        self.video_source = None
        self.mediaLabel.clear()

   
    def importButtonEvent(self):
        try:
            if self.inspectionDetailsFlag == False:
                raise Exception("Missing: Inspection Details")
            
            if self.videoAvailable == True: #handle subsequent import of video
               self.paused = False
               self.playPauseVideoEvent()

            if self.fileType.currentIndex() == 0: 
                self.open_video()
            else: #clear video capture initialization before importing image
                if self.cap:
                  self.cap.release()
                  self.total_frames = 0
                  self.fps = 0
                  self.current_frame = 0
                if self.timer:
                   self.timer.stop()
                self.open_image()
    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"{e}")
            return


    def open_image(self):
        try:
            self.image_source, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Image Files (*.jpg *.png)")

            if self.image_source:
                self.imageAvailable = True
                self.scrubberBar.setVisible(False)
                self.videoWidgets.setVisible(False)

                # Load the image and display it in the label
                self.image_pixmap = QPixmap(self.image_source)
                scaled_pixmap = self.image_pixmap.scaled(self.mediaLabel.size(), Qt.AspectRatioMode.KeepAspectRatio)
                self.mediaLabel.setPixmap(scaled_pixmap)
                self.mediaLabel.setAlignment(Qt.AlignCenter)

        except Exception as e:
            self.imageAvailable = False
            QMessageBox.warning(self, "Error", f"{e}")
            return


    def open_video(self):
        try:
            self.video_source, _ = QFileDialog.getOpenFileName(
                self, "Open Video", "", "Video Files (*.mp4 *.avi *.wmv)")
    
            if self.video_source:
                self.scrubberBar.setVisible(True)
                self.videoWidgets.setVisible(True)
                self.videoAvailable = True

                # Load the video and display it in the label
                self.cap = cv2.VideoCapture(self.video_source)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                self.scrubberBar.setRange(0, self.total_frames - 1)
                self.scrubberBar.setTickInterval(int(self.total_frames / 10))
                self.scrubberBar.setValue(0)

                self.timer.setTimerType(Qt.PreciseTimer)
                self.timer.timeout.connect(self.play_video)
                
                self.speedSlider.valueChanged.connect(self.updateSpeed) #video play event handler method
        
                # Calculate the initial interval of timer based on the value of speedSlider
                ratio = self.speedSlider.value() / (self.speedSlider.maximum() - self.speedSlider.minimum())
                self.timer.setInterval(int((1000 // self.fps) / ratio))

                if self.paused == True: 
                    self.playPauseVideoEvent()
                self.timer.start()

                self.updateMediaLabel()
        except Exception as e:
            self.scrubberBar.setVisible(False)
            self.videoWidgets.setVisible(False)
            self.videoAvailable = False
            QMessageBox.warning(self, "Error", f"{e}")
            return
        
  
    def play_video(self):
        new_frame = self.scrubberBar.value() + 1

        if new_frame >= self.total_frames:
            self.timer.stop()
            icon = QIcon()
            icon.addFile(u"icons/start.png", QSize(), QIcon.Normal, QIcon.Off)
            self.playVidButton.setIcon(icon)
            self.current_frame = 0
            self.scrubberBar.setValue(self.current_frame)
            self.paused = False
            self.playPauseVideoEvent()
        else:
            self.current_frame = new_frame
            self.scrubberBar.setValue(self.current_frame)    
        self.updateMediaLabel()


    def updateMediaLabel(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            success = self.cap.grab()

            # Retrieve the grabbed frame
            if success:
                _, frame = self.cap.retrieve()
            
                # Perform further processing with the frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qimage = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                scaled_qimage = qimage.scaled(self.mediaLabel.width(), self.mediaLabel.height(), Qt.KeepAspectRatio)
                self.mediaLabel.setPixmap(QPixmap.fromImage(scaled_qimage))
                self.mediaLabel.setAlignment(Qt.AlignCenter)


    def updateSpeed(self, value):
        # Calculate the new interval of timer based on the value of speedSlider
        ratio = value / (self.speedSlider.maximum() - self.speedSlider.minimum())
        new_interval = int((1000 // self.fps) / ratio)
        self.timer.setInterval(new_interval)


    def skipFootageEvent(self):
        new_frame = self.scrubberBar.value() + self.fps * 10
        self.current_frame = int(min(new_frame, self.total_frames - 1))
        self.scrubberBar.setValue(self.current_frame)
        self.updateMediaLabel()


    def backFootageEvent(self):
        new_frame = self.scrubberBar.value() - self.fps * 10
        self.current_frame = int(max(new_frame, 0))
        self.scrubberBar.setValue(self.current_frame)
        self.updateMediaLabel()


    def playPauseVideoEvent(self):
        icon = QIcon()
        try:
            if not self.paused:
               self.timer.stop()
               icon.addFile(u"icons/start.png", QSize(), QIcon.Normal, QIcon.Off)
               self.playVidButton.setIcon(icon)
               self.playVidButton.setIconSize(QSize(60, 60))
               self.paused = True
            else:
               self.timer.start()
               icon.addFile(u"icons/pause.png", QSize(), QIcon.Normal, QIcon.Off)
               self.playVidButton.setIcon(icon)
               self.playVidButton.setIconSize(QSize(60, 60))
               self.paused = False
        except Exception as e:
            QMessageBox.warning(self, "Error", f"{e}")
            return


    def captureButtonEvent(self):
        if self.imageAvailable == False and self.videoAvailable == False: 
            QMessageBox.warning(self, "Invalid Action", "Missing media file")
            return
        if self.fileType.currentIndex() == 0:
            if not self.paused:
                self.playPauseVideoEvent()

        self.label = QLabel()
        # Create a QDialog to show the captured frame
        self.frameDialog = QDialog(self)
        self.frameDialog .setWindowTitle("Captured Frame")
        self.frameDialog .setFixedSize(900, 500)
        self.frameDialog .setStyleSheet("background-color: #575863; color: white;")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.frameDialog .setLayout(layout)

        # Create the checkbox and spinbox pairs
        self.label1 = QLabel("Mahogany Leaves")
        self.spinbox1 = QSpinBox()
        self.spinbox1.setStyleSheet("background-color: #ffffff; color: #000000")
        self.label2 = QLabel("Mahogany Fruit")
        self.spinbox2 = QSpinBox()
        self.spinbox2.setStyleSheet("background-color: #ffffff; color: #000000")
        self.label3 = QLabel("Mahogany Branches")
        self.spinbox3 = QSpinBox()
        self.spinbox3.setStyleSheet("background-color: #ffffff; color: #000000")
        
        # Create a QHBoxLayout for each checkbox-spinbox pair and add them to the layout
        layout1 = QHBoxLayout()
        layout1.addWidget(self.label1)
        layout1.addWidget(self.spinbox1)
        layout1.addWidget(self.label2)
        layout1.addWidget(self.spinbox2)
        layout1.addWidget(self.label3)
        layout1.addWidget(self.spinbox3)
        layout.addLayout(layout1)

        if self.fileType.currentIndex() == 1:  #import image
            frame = cv2.imread(self.image_source)
        else:
            if self.cap:
                # Capture the current frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
    
        # Convert the frame to a QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Resize the QImage to fit the dialog label
        scaled_qimage = qimage.scaled(self.frameDialog .width(), self.frameDialog .height(),
                            Qt.KeepAspectRatio)
        # Create a QLabel to display the captured frame
        self.label.setPixmap(QPixmap.fromImage(scaled_qimage))
        # Create a QPushButton for saving the captured frame
        save_button = QPushButton('Save Frame')
        save_button.setStyleSheet("background-color: #abacb1; color: #575863; width:100px; height:50px; font: 57 12pt Dosis Medium;")
        save_button.clicked.connect(lambda: self.saveCapturedFrame())
        layout.addWidget(save_button)
        # Center align the checkboxes and spinboxes layout
        layout.setAlignment(Qt.AlignCenter)
        self.frameDialog.exec_()

        if self.fileType.currentIndex() == 0:
            self.playPauseVideoEvent()


    def saveCapturedFrame(self):
        self.location = self.location_text_edit.toPlainText()
        self.serialnum = self.serialnum_text_edit.toPlainText()
        filename = "{}_{}_capture{}.png".format(self.location, self.serialnum, datetime.now().strftime("%H%M%S"))
    
        painter = QPainter(self.label.pixmap()) #Draw on the QImage
        painter.setPen(QPen(Qt.black))
        painter.setFont(QFont("Arial", 8))
    
        text_position = QPoint(10, 18)
    
        text = """Mahogany Leaves: {}
                  Mahogany Fruits: {}
                  Mahogany Branches: {}""".format(self.spinbox1.value(), self.spinbox2.value(), self.spinbox3.value())
        painter.drawText(text_position, text)
        painter.end() #End painting
        path = "{}/{}".format(self.directory, filename)
        self.label.pixmap().toImage().save(path) #Save the QImage
    
        QMessageBox.information(self, "Saved", "Captured frame saved successfully")

        self.frameDialog.reject()


    def load_model(self): #load model CPU only
        try:
            temp_model, _ = QFileDialog.getOpenFileName(self, 'Load YOLOv5 Model', '',
                                            "PyTorch Checkpoint (*.pt)")
            if temp_model:
                self.model = DetectMultiBackend(temp_model, device=select_device('cpu'), dnn=self.dnn,
                                                data=self.data, fp16=self.half)
                self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
                self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
                if temp_model: self.modelLoaded = True  #signal that model has been loaded
                self.modelStatus.setVisible(True)
                self.modelStatus.setText(f'Model Loaded: {os.path.basename(temp_model)}')
            
        except Exception as e:
            self.modelStatus.setVisible(False)
            QMessageBox.warning(self, "Error", f"{e}")
            return


    def runButtonEvent(self):
        try:
            if self.modelLoaded == False: raise Exception("Model not loaded")
            if self.fileType.currentIndex() == 0 and not self.videoAvailable and self.predictionThreadFinished: raise Exception("No Video Available")
            if self.fileType.currentIndex() == 1 and not self.imageAvailable and self.predictionThreadFinished: raise Exception("No Image Available")
            # If prediction thread is currently idle, set up to start
            if self.predictionThreadFinished:
                self.predictionThread = None
                self.predictionThread = threading.Thread(target=self.runPrediction)
                self.predictionThread.daemon = True
                self.predictionThreadFinished = False
                if self.fileType.currentIndex() == 0:
                    if not self.paused:
                        self.playPauseVideoEvent()
            
            # Start prediction thread if from idle state, else forcefully finish the thread
            if not self.predictionThread.is_alive() or self.predictionThread is None:
                self.predictionThread.start()
                self.predictionThreadFinished = False
                self.runButton.setText(f'Stop Processing')
            else:
                self.predictionThreadFinished = True
                self.runButton.setText(f'Run Detection')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"{e}")

    # Target function for the prediction thread, this function will start the prediction for the selected image/video
    def runPrediction(self):
        try:
            cap = None
            video_writer = None
            newHeight = 416
            newWidth = 416
            current_frame = 0
            ret = None
            isImage = False

            if self.fileType.currentIndex() == 1:
                image_file_name = os.path.splitext(os.path.basename(self.image_source))[0]
                filename = "{}_{}_{}_detection.png".format(self.location, self.serialnum, image_file_name)
                file_path = os.path.join(self.directory, filename)
                frame = cv2.imread(self.image_source)
                total_frames = 1
                isImage = True
            else:
                video_file_name = os.path.splitext(os.path.basename(self.video_source))[0]
                filename = "{}_{}_{}_detection.mp4".format(self.location, self.serialnum, video_file_name)
                file_path = os.path.join(self.directory, filename)
                cap = cv2.VideoCapture(self.video_source)
                ret, frame = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(file_path, fourcc, fps, (newWidth, newHeight))

            height, width, _ = frame.shape
            
            # Write each frame to the video write
            while current_frame < total_frames and not self.predictionThreadFinished:
                if ret or isImage:
                    resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)
                    bs = 1  # batch_size
                    self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
                    
                    im = letterbox(resized_frame, (416, 416), stride=32, auto=self.pt)[0]
                    im = im.transpose((2, 0, 1))[::-1]
                    im = np.ascontiguousarray(im)
                    im = torch.from_numpy(im).to(self.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
    
                    # Inference
                    self.pred = self.model(im, augment=self.augment, visualize=self.visualize)
    
                    # NMS
                    self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

                    # Process predictions
                    for i, det in enumerate(self.pred):  # per image
                        im0 = resized_frame.copy()
                        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
    
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                                
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                else: pass

                if self.fileType.currentIndex() == 0:
                    # Write the resized frame to the video writer
                    video_writer.write(im0)
                    ret, frame = cap.read()

                if self.fileType.currentIndex() == 1:
                    # Save the processed photo to file_path
                    cv2.imwrite(file_path, im0)
                    original_image = cv2.resize(im0, (width, height), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(file_path, original_image)
                
                current_frame = current_frame + 1

            if cap:
                # Release the video capture and video writer
                cap.release()
                video_writer.release()
            self.runButton.setText(f'Run Detection')
            self.predictionThreadFinished = True
            print('Prediction thread is finished running')
            return

        except Exception as e:
            if cap is not None:
                if cap.isOpened(): 
                    cap.release()
            if video_writer is not None:
                if video_writer.isOpened():
                    video_writer.release()
            self.runButton.setText(f'Run Detection')
            self.predictionThreadFinished = True
            print(f'Prediction thread encountered an error: {e}')
            return  


    def handleTabChange(self):
        if self.widget1.currentIndex() == 2:
             if self.cameraComboBox.currentIndex() == 1:
                self.cameraDropdown.setVisible(False)
                self.ipAddrText.setVisible(True)
                self.selectCameraLabel.setText("IP Address of DJI Mini 2 Control Server")
             else:
                self.ipAddrText.setVisible(False)
                self.selectCameraLabel.setText("Select Camera")
                self.cameraDropdown.setVisible(True)
        else:
            self.stopStreamEvent()


    def setCamera(self, index):
        try:
            self.camera = cv2.VideoCapture(index)
            if not self.camera.isOpened():
                raise Exception('Camera could not connect to the source')
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.camera.setExceptionMode(True)
            self.cameraConnected = True
        except Exception as e:
            self.cameraConnected = False
            QMessageBox.warning(self, "Error setting the camera", f"{e}")


    def update_image(self):
        streamInterrupted = False
        while self.isStreaming and not streamInterrupted:
            try:
                ret, frame = self.camera.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Apply the brightness, saturation, and contrast adjustments to the frame
                    brightness = self.brightnessSlider.value()
                    saturation = self.saturationSlider.value()
                    contrast = self.contrastSlider.value()
                    
                    frame = cv2.convertScaleAbs(frame, alpha=(contrast/255.0+1), beta=brightness) #127.0
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    h, s, v = cv2.split(frame)
                    s = cv2.add(s, saturation)
                    frame = cv2.merge((h, s, v))
                    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
                    frame = cv2.GaussianBlur(frame, (3,3), 0) 

                    # if recording button is toggled, start recording
                    if self.is_recording:
                        recframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if self.video_writer.isOpened():
                            self.video_writer.write(recframe)

                    size = self.streamLabel.size()
                    # Convert the frame to a QImage and display it in the window
                    self.image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    self.streamLabel.setPixmap(QPixmap.fromImage(self.image).scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            
            except Exception as e:
                print(f"Stream Interrupted: {e}")        
                streamInterrupted = True
                self.stopStreamEvent(fromThread=True)
                return
        return
    

    def recordStreamEvent(self, fromThread=False):
        try:
            if self.isStreaming == False: raise Exception("Start stream first")

            else:
                icon = QIcon()
                self.streamDirectory_text_edit = self.findChild(QtWidgets.QTextEdit, "streamDirectory")
                video_path = rf"{self.streamDirectory_text_edit.toPlainText()}"
                video_path = os.path.join(video_path, 'Recordings and Captures')
    
                if not self.is_recording:
                    # Start recording
                    self.timer1.timeout.connect(self.update_elapsed_time)
                    self.videoFilename = ""
        
                    # Create a dialog box to input the filename
                    filename, ok = QtWidgets.QInputDialog.getText(self, "Video recording", "Enter filename:", QtWidgets.QLineEdit.Normal, self.videoFilename)
                    if ok and filename:
                        self.videoFilename = f"{filename}.wmv"
                    else: 
                        return
        
                    file_path = os.path.join(video_path, self.videoFilename)
                    frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Create a VideoWriter object
                    fps = int(self.camera.get(cv2.CAP_PROP_FPS))
                    self.video_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'WMV2'), fps, (frame_width, frame_height))

                    self.recordTime.setVisible(True)
                    self.elapsedTime = QDateTime.currentDateTime()

                    self.is_recording = True
                    self.timer1.start(1000)  # Update elapsed time every second
                    icon.addFile(u"icons/end.png", QSize(40,40), QIcon.Normal, QIcon.Off)
                    self.recordStream.setIcon(icon)
                    self.recordStream.setStyleSheet(u"QPushButton {\n""border-radius: 40px;\n" "width: 80px;\n"
                                                    "height: 80px;\n" "background-color:  rgb(87, 87, 99);\n""}")
        
                else:
                    # Stop recording
                    self.video_writer.release()
                    self.is_recording = False
                    self.recordTime.setVisible(False)
                    self.timer1.stop()  # Stop updating elapsed time
        
                    icon.addFile(u"icons/record.png", QSize(40,40), QIcon.Normal, QIcon.Off)
                    self.recordStream.setIcon(icon)
                    self.recordStream.setStyleSheet(u"QPushButton {\n""border-radius: 40px;\n" "width: 80px;\n"
                                                    "height: 80px;\n" "background-color: red;\n""}")
            
                    if not fromThread:
                        QMessageBox.information(self, "Recording Saved", f"{self.videoFilename} Saved in {video_path}")

        except Exception as e:
            if not fromThread:
                QMessageBox.warning(self, "Problem encountered with recording attmempt", f"{e}")


    def update_elapsed_time(self):
        elapsed = self.elapsedTime.secsTo(QDateTime.currentDateTime())
        self.recordTime.setText(QCoreApplication.translate("ImportWindow_UI", u"Recording time: " + f"{QTime(0, 0).addSecs(elapsed).toString('hh:mm:ss')}", None))


    def captureStreamEvent(self):
        try:
            if self.isStreaming == False: raise Exception("Start stream first") 
            self.streamDirectory_text_edit = self.findChild(QtWidgets.QTextEdit, "streamDirectory")
            image_path = rf"{self.streamDirectory_text_edit.toPlainText()}"
            image_path = os.path.join(image_path, 'Recordings and Captures')
            self.capturedFilename = ""

            ret, frame = self.camera.read()
            if ret:
                # Create a dialog box to input the filename
                filename, ok = QtWidgets.QInputDialog.getText(self, "Frame captured", "Enter filename:", QtWidgets.QLineEdit.Normal, self.capturedFilename)
                
                if ok and filename: self.capturedFilename = f"{filename}.png"
                else:
                    return
                cv2.imwrite(os.path.join(image_path, self.capturedFilename), frame)

                QtWidgets.QMessageBox.information(self, "Saved Capture", f"{image_path}")   #Show a message box to confirm the frame capture
        except Exception as e:
            QMessageBox.warning(self, "Problem encountered with frame capture", f'{e}')


    def stopStreamEvent(self, fromThread=False):
        try:
            #handle stop event if stop is triggered
            if self.isStreaming == False and self.widget1.currentIndex() == 2: 
                raise Exception("Start stream first")
            
            #handle recording if stop is triggered 
            if self.is_recording: 
                self.recordStreamEvent(fromThread)

            #handle streaming if stop is triggered    
            self.isStreaming = False
            if self.client is not None:
                self.client.stopLiveStream()
                self.client = None
            if self.camera is not None:
                if self.camera.isOpened():
                    self.camera.release()
            self.streamLabel.clear()
            self.clearStream()

        except Exception as e:
            if not fromThread:
                QMessageBox.warning(self, "Stream Unavailable", f'{e}')


    def clearStream(self):
        self.startStream.setProperty("cursor", QCursor(Qt.PointingHandCursor))
        self.startStream.setStyleSheet(u"QPushButton {\n" "border-radius: 8px;\n" """font: 57 12pt "Dosis Medium";\n"""
                                                        "color: rgb(255, 255, 255);\n" "background-color: #AA0000;\n}"
                                        "QPushButton::Hover {\n" "background-color: rgb(255, 0, 0);\n}")
        self.startStream.setEnabled(True)
        self.streamLabel.clear()
    

    def startStreamEvent(self):
        if self.cameraComboBox.currentIndex() == 1: #RTMP mode live streaming (check if available ang connection)
            ip = self.ip_text_edit.toPlainText()
            url = socket.gethostbyname(socket.gethostname())
            rtmpAddr = f'rtmp://{url}/live/stream'
            if ip == "": 
                ip = self.defIP 
            try:
                self.client = ClientRequests(ip)
                self.client.startLiveStream(url)
            except:
                QMessageBox.warning(self, "Error with connection to control server", 
                                    "Program will try to continue RTMP with available video streams connected in NGINX")
            self.setCamera(rtmpAddr)

        else: self.setCamera(self.cameraDropdown.currentIndex()) #Device camera mode live stream
        
        if self.cameraConnected:
            self.startStream.setProperty("cursor", QCursor(Qt.ForbiddenCursor))
            self.startStream.setStyleSheet(u"QPushButton {\n" "border-radius: 8px;\n" """font: 57 12pt "Dosis Medium";\n"""
                                                        "color: rgb(255, 255, 255);\n" "background-color: rgb(39, 39, 54);\n}")
            self.startStream.setEnabled(False)

            self.isStreaming = True
            self.streamThread = threading.Thread(target=self.update_image)
            self.streamThread.daemon = True
            self.streamThread.start()


    def MoveWindow(self, event):
        if self.isMaximized() == False:
            self.move(self.pos() + event.globalPos() - self.clickPosition)
            self.clickPosition = event.globalPos()
            event.accept()
            pass


    def mousePressEvent(self, event):
        self.clickPosition = event.globalPos()


class SplashScreen(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        uic.loadUi("SplashScreen.ui", self)  #load ui file
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) #remove default title bar

    def Progress(self):
        initial_value = self.progressBar.value()
        for i in range(initial_value, 100):
            time.sleep(0.1)
            self.progressBar.setValue(i)


def show_main_window(window):
    App = window()
    App.show()


if __name__ == "__main__":
    MainApp = QtWidgets.QApplication(sys.argv)

    SplashScreen = SplashScreen()
    SplashScreen.show()
    SplashScreen.Progress()

    QtCore.QTimer.singleShot(1000, SplashScreen.close)
    QtCore.QTimer.singleShot(50, lambda: show_main_window(MainWindow))

    
    os.system(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), 'launch_nginx_batch.vbs'))
    os.system(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), 'launch_nginx.vbs'))

    sys.exit(MainApp.exec_())