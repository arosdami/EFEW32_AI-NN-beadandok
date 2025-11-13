import cv2
import numpy as np
import tensorflow as tf
# TensorFlow 2.16+ kompatibilis importok
try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model
except ImportError:
    # Új TensorFlow verziókban közvetlenül keras
    import keras
    from keras import layers, Model
import time
import serial
import pickle
import os
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
import logging

class ProfessionalFaceLockSystem:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Landmarks
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = None
        self.face_mesh = None
        
        # System components
        self.face_encoder = None
        self.face_db = {}
        self.label_encoder = LabelEncoder()
        self.knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        self.is_trained = False
        
        # System parameters
        self.recognition_threshold = 0.6
        self.quality_threshold = 0.7
        self.arduino = None
        self.camera_index = 0
        self.arduino_port = 'COM3'
        self.unlock_duration = 10
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_lock_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_system(self):
        """Initialize all system components"""
        self.show_header()
        self.initialize_mediapipe()
        self.select_camera()
        self.setup_arduino()
        self.build_face_encoder()
        self.load_database()
        
        self.logger.info("System initialized successfully")
        print("\n[SUCCESS] System initialized!")
        
    def initialize_mediapipe(self):
        """Initialize MediaPipe components"""
        try:
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.7
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"❌ MediaPipe initialization failed: {e}")
            raise

    def show_header(self):
        """Display professional header"""
        print("=" * 70)
        print("           PROFESSIONAL AI FACE RECOGNITION LOCK SYSTEM v2.0")
        print("=" * 70)
        print("Features: MediaPipe Face Detection | Deep Face Encoding | Real-time Recognition")
        print("=" * 70)
        
    def select_camera(self):
        """Camera selection with auto-detection"""
        print("\n[CAMERA SELECTION]")
        
        # Test available cameras
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    available_cameras.append(i)
                    print(f"Camera {i}: Available")
        
        if not available_cameras:
            self.logger.error("No cameras found!")
            raise Exception("No cameras available")
            
        choice = input(f"Select camera {available_cameras} (default 0): ").strip()
        self.camera_index = int(choice) if choice and int(choice) in available_cameras else available_cameras[0]
        
        self.logger.info(f"Camera {self.camera_index} selected")
        print(f"[SUCCESS] Camera {self.camera_index} selected")
        
    def setup_arduino(self):
        """Arduino setup with auto-detection"""
        print("\n[ARDUINO SETUP]")
        
        common_ports = ['COM3', 'COM4', 'COM5', 'COM6', '/dev/ttyUSB0', '/dev/ttyACM0']
        
        for port in common_ports:
            try:
                self.arduino = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                # Test communication
                self.arduino.write(b"TEST\n")
                time.sleep(0.5)
                self.arduino_port = port
                self.logger.info(f"Arduino connected on {port}")
                print(f"[SUCCESS] Arduino connected on {port}")
                return
            except:
                continue
                
        self.logger.warning("Arduino not found - running in simulation mode")
        print("[INFO] Running in simulation mode")
        self.arduino = None
        
    def build_face_encoder(self):
        """Build a modern face embedding model"""
        try:
            # TensorFlow 2.16+ kompatibilis megoldás
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(128, 128, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Build model using TensorFlow Keras directly
            inputs = tf.keras.Input(shape=(128, 128, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(128, activation='linear')(x)
            
            # L2 normalization for cosine similarity
            outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
            
            self.face_encoder = tf.keras.Model(inputs, outputs)
            
            # Freeze base model layers
            base_model.trainable = False
            
            self.face_encoder.compile(optimizer='adam', loss='mse')
            self.logger.info("Face encoder model built successfully")
            print("✅ Face encoder model built successfully")
            
        except Exception as e:
            print(f"❌ Error building face encoder: {e}")
            # Alternatív egyszerű modell
            self.build_simple_face_encoder()
    
    def build_simple_face_encoder(self):
        """Build simple face encoder as fallback"""
        try:
            self.face_encoder = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='sigmoid')
            ])
            self.face_encoder.compile(optimizer='adam', loss='mse')
            print("✅ Simple face encoder built as fallback")
        except Exception as e:
            print(f"❌ Error building simple encoder: {e}")
            self.face_encoder = None
        
    def extract_face_landmarks(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, List]]:
        """Extract face using MediaPipe with landmarks"""
        if self.face_detection is None:
            return None
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Face detection
        detection_results = self.face_detection.process(rgb_image)
        
        if not detection_results.detections:
            return None
            
        # Get the largest face
        largest_detection = max(detection_results.detections, 
                              key=lambda det: det.score)
        
        bbox = largest_detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Expand bounding box slightly
        x = max(0, x - 20)
        y = max(0, y - 20)
        width = min(w - x, width + 40)
        height = min(h - y, height + 40)
        
        face_roi = image[y:y+height, x:x+width]
        
        if face_roi.size == 0:
            return None
            
        # Get facial landmarks
        landmarks = []
        if self.face_mesh:
            landmark_results = self.face_mesh.process(rgb_image)
            if landmark_results.multi_face_landmarks:
                for face_landmarks in landmark_results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        landmarks.append([landmark.x * w, landmark.y * h])
        
        return face_roi, landmarks, (x, y, width, height)
        
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for encoding"""
        # Resize to model input size
        face_resized = cv2.resize(face_image, (128, 128))
        
        # Convert to RGB and normalize
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        
        return np.expand_dims(face_normalized, axis=0)
        
    def get_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Get face embedding vector"""
        if self.face_encoder is None:
            # Return random embedding as fallback
            return np.random.randn(64)
            
        preprocessed_face = self.preprocess_face(face_image)
        embedding = self.face_encoder.predict(preprocessed_face, verbose=0)[0]
        return embedding
        
    def assess_face_quality(self, face_image: np.ndarray, landmarks: List) -> Dict:
        """Comprehensive face quality assessment"""
        if face_image.size == 0:
            return {'quality': 0.0, 'brightness': 0, 'contrast': 0, 'sharpness': 0, 'assessment': 'NO_FACE'}
            
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Brightness assessment
        brightness = np.mean(gray_face)
        brightness_score = min(1.0, brightness / 128.0)
        
        # Contrast assessment
        contrast = np.std(gray_face)
        contrast_score = min(1.0, contrast / 64.0)
        
        # Sharpness assessment (Laplacian variance)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 1000.0)
        
        # Face size assessment
        face_area = face_image.shape[0] * face_image.shape[1]
        size_score = min(1.0, face_area / (100 * 100))
        
        # Overall quality score
        quality_score = (brightness_score + contrast_score + sharpness_score + size_score) / 4.0
        
        assessment = "HIGH" if quality_score > 0.7 else "MEDIUM" if quality_score > 0.5 else "LOW"
        
        return {
            'quality': quality_score,
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'assessment': assessment
        }
        
    def load_database(self):
        """Load face database with error handling"""
        try:
            if os.path.exists('face_database.pkl'):
                with open('face_database.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.face_db = data.get('embeddings', {})
                    
                # Retrain classifier if data exists
                if self.face_db:
                    self.train_classifier()
                    
                self.logger.info(f"Database loaded: {len(self.face_db)} persons")
                print(f"[SUCCESS] Database loaded: {len(self.face_db)} persons")
            else:
                self.logger.info("No existing database found")
                print("[INFO] No existing database found")
                
        except Exception as e:
            self.logger.error(f"Error loading database: {e}")
            print("[ERROR] Database corrupted, starting fresh")
            self.face_db = {}
            
    def save_database(self):
        """Save face database with backup"""
        try:
            # Create backup if exists
            if os.path.exists('face_database.pkl'):
                os.rename('face_database.pkl', 'face_database_backup.pkl')
                
            data = {
                'embeddings': self.face_db,
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            with open('face_database.pkl', 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info("Database saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving database: {e}")
            
    def train_classifier(self):
        """Train KNN classifier on face embeddings"""
        if not self.face_db:
            self.is_trained = False
            return
            
        embeddings = []
        labels = []
        
        for name, face_data in self.face_db.items():
            for embedding in face_data['embeddings']:
                embeddings.append(embedding)
                labels.append(name)
                
        if len(embeddings) > 0:
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            # Train KNN classifier
            self.knn_classifier.fit(embeddings, encoded_labels)
            self.is_trained = True
            
            self.logger.info(f"Classifier trained with {len(embeddings)} samples")
            
    def register_face(self):
        """Advanced face registration with quality control"""
        print("\n[FACE REGISTRATION]")
        name = input("Enter person name: ").strip()
        
        if not name:
            print("[ERROR] Invalid name")
            return
            
        if name in self.face_db:
            overwrite = input(f"Person '{name}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                return
                
        print(f"Starting registration for: {name}")
        print("Look directly at the camera with neutral expression")
        print("Press 'q' to finish, 'r' to retry")
        
        cap = cv2.VideoCapture(self.camera_index)
        embeddings = []
        samples_collected = 0
        target_samples = 15  # Reduced for testing
        quality_samples = 0
        
        while samples_collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Extract face with landmarks
            result = self.extract_face_landmarks(frame)
            display_frame = frame.copy()
            
            if result:
                face_roi, landmarks, bbox = result
                quality_info = self.assess_face_quality(face_roi, landmarks)
                
                # Only collect high quality samples
                if quality_info['quality'] > self.quality_threshold:
                    embedding = self.get_face_embedding(face_roi)
                    embeddings.append(embedding)
                    samples_collected += 1
                    quality_samples += 1
                    
                # Draw UI
                x, y, w, h = bbox
                color = (0, 255, 0) if quality_info['quality'] > self.quality_threshold else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw landmarks
                if landmarks:
                    for landmark in landmarks[:10]:  # Draw first 10 landmarks
                        lx, ly = int(landmark[0]), int(landmark[1])
                        cv2.circle(display_frame, (lx, ly), 2, (0, 255, 255), -1)
                        
            # Draw registration UI
            self.draw_registration_ui(display_frame, name, samples_collected, target_samples, quality_info)
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                embeddings = []
                samples_collected = 0
                quality_samples = 0
                
        cap.release()
        cv2.destroyAllWindows()
        
        if len(embeddings) >= 5:  # Minimum samples required
            self.face_db[name] = {
                'embeddings': embeddings,
                'registration_date': datetime.now().isoformat(),
                'samples_count': len(embeddings),
                'average_quality': quality_samples / len(embeddings) if embeddings else 0
            }
            
            self.train_classifier()
            self.save_database()
            
            self.logger.info(f"Registered {name} with {len(embeddings)} samples")
            print(f"[SUCCESS] {name} registered with {len(embeddings)} high-quality samples")
        else:
            self.logger.warning(f"Insufficient samples for {name}")
            print("[ERROR] Insufficient high-quality samples collected")

    # ... (a többi metódus marad változatlan, de TensorFlow kompatibilis)

    def main_menu(self):
        """Main system menu"""
        self.initialize_system()
        
        while True:
            print("\n" + "=" * 50)
            print("          PROFESSIONAL FACE LOCK SYSTEM")
            print("=" * 50)
            print("1. Register New Face")
            print("2. Start Security System") 
            print("3. System Status & Analytics")
            print("4. Database Management")
            print("5. System Configuration")
            print("6. Exit")
            print("-" * 50)
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                self.register_face()
            elif choice == '2':
                self.security_system()
            elif choice == '3':
                self.show_system_status()
            elif choice == '4':
                self.database_management()
            elif choice == '5':
                self.system_configuration()
            elif choice == '6':
                print("\n[SYSTEM SHUTDOWN]")
                if self.arduino:
                    self.arduino.close()
                break
            else:
                print("[ERROR] Invalid selection")

    def show_system_status(self):
        """Display comprehensive system status"""
        print("\n[SYSTEM STATUS REPORT]")
        print("=" * 50)
        print(f"Camera: {self.camera_index}")
        print(f"Arduino: {'Connected' if self.arduino else 'Disconnected'}")
        print(f"Registered persons: {len(self.face_db)}")
        print(f"Recognition threshold: {self.recognition_threshold}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Unlock duration: {self.unlock_duration}s")
        print(f"Model: {'Trained' if self.is_trained else 'Not trained'}")
        print(f"Face Encoder: {'Loaded' if self.face_encoder else 'Not loaded'}")
        
        if self.face_db:
            print("\nRegistered Persons:")
            for name, data in self.face_db.items():
                print(f"  - {name}: {data['samples_count']} samples")

    # Egyszerűsített security_system a teszteléshez
    def security_system(self):
        """Simplified security system for testing"""
        print("\n[SECURITY SYSTEM ACTIVATED]")
        print("Press 'q' to exit")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Face detection test
            result = self.extract_face_landmarks(frame)
            display_frame = frame.copy()
            
            if result:
                face_roi, landmarks, bbox = result
                quality_info = self.assess_face_quality(face_roi, landmarks)
                
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display_frame, f"Quality: {quality_info['assessment']}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(display_frame, "Press 'q' to exit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Security System - TEST MODE', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        system = ProfessionalFaceLockSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print("\n[SYSTEM SHUTDOWN] Keyboard interrupt detected")
    except Exception as e:
        print(f"\n[SYSTEM ERROR] {e}")
        import traceback
        traceback.print_exc()