import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageClassifier(QDialog):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        uic.loadUi("D:\Programming\Data Science\Moamen\Computer Vision\Git_Repo\Amit_AI_62\Computer_vision_project\GUI.ui", self)

        # Load model
        self.model = load_model("D:\Programming\Data Science\Moamen\Computer Vision\Git_Repo\Amit_AI_62\Computer_vision_project\my_model.h5")

        self.vgg_model = VGG16(weights='imagenet', include_top=False)
        
        self.ensembels = []
        for i in range(10):
            filename = 'model_' + str(i + 1) + '.h5'
            filepath = "D:\Programming\Data Science\Moamen\Computer Vision\Git_Repo\Amit_AI_62\Computer_vision_project\Ensembels\\" + filename
            ensemble = load_model(filepath)
            self.ensembels.append(ensemble)
        
        # Load buttons
        self.load_button_1 = self.findChild(QPushButton, 'browseButton')
        self.load_button_1.clicked.connect(self.load_image)

        self.load_button_2 = self.findChild(QPushButton, 'predictButton')
        self.load_button_2.clicked.connect(self.predict_with_model)

        
        self.load_button_3 = self.findChild(QPushButton, 'ensembleButton')
        self.load_button_3.clicked.connect(self.predic_with_ensemble)

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'imageView')
        self.scene = QGraphicsScene(self)
        self.image_viewer_1.setScene(self.scene)

        # Result labels
        self.result_label = self.findChild(QLabel, 'categoryLabel')
        self.result_label_2 = self.findChild(QLabel, 'confidenceLabel')

        # Class names mapping
        self.classes = {0 : 'buildings' ,1 : 'forest',2 : 'glacier',3 : 'mountain',4 : 'sea',5 : 'street'}

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            try:
                # Load and resize image for viewing
                img = cv2.imread(self.image_path)
                img_resized = cv2.resize(img, (224, 224))  # Resize for display
                height, width, _ = img_resized.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                self.scene.clear()
                self.scene.addPixmap(QPixmap.fromImage(q_img))
                self.image_viewer_1.setScene(self.scene)
            except Exception as e:
                self.result_label.setText(f"Error loading image: {e}")

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150,150))  # Resize to model input size
            img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            self.result_label.setText(f"Error processing image: {e}")
            return None

    def predict_with_model(self):
        if self.image_path:
            img = self.preprocess_image(self.image_path)
            if img is not None:
                try:
                    predictions = self.model.predict(img)

                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100

                    # Update labels with class names and confidence
                    self.result_label.setText(f'Class: {self.classes[class_idx]}')
                    self.result_label_2.setText(f'Confidence: {confidence:.2f}%')
                except Exception as e:
                    self.result_label.setText(f"Prediction error: {e}")
        else:
            self.result_label.setText("Please load an image first.")
    

    def extract_vgg_features(self, image_path):
        try:
            # Load and preprocess image for VGG16
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150, 150))
            img = preprocess_input(img)  # VGG16 specific preprocessing
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Extract features
            features = self.vgg_model.predict(img)
            return features
        except Exception as e:
            self.result_label.setText(f"Error extracting features: {e}")
            return None
        
    def predic_with_ensemble(self):
        if self.image_path:
            try:
                features = self.extract_vgg_features(self.image_path)
                if features is not None:
                    predictions = []
                    for model in self.ensembels:
                        pred = model.predict(features)
                        predictions.append(pred)
                
                    predictions = np.array(predictions)
                    avg_predictions = np.mean(predictions, axis= 0)

                    class_idx = np.argmax(avg_predictions[0])
                    confidence = np.max(avg_predictions[0]) * 100

                    # Update labels with class names and confidence
                    self.result_label.setText(f'Class: {self.classes[class_idx]}')
                    self.result_label_2.setText(f'Confidence: {confidence:.2f}%')
            except Exception as e:
                self.result_label.setText(f"Prediction error: {e}")
        else:
            self.result_label.setText("Please load an image first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())
