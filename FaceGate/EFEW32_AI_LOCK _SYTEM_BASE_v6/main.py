import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import serial
import pickle
import os
import sys
from pathlib import Path
import hashlib
import json
import base64

class ProfessionalFaceGate:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.cnn_model = None
        try:
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                self.lbph_recognizer = None
        
        self.face_db = {}
        self.lbph_trained = False
        
        self.threshold = 0.75
        self.arduino = None
        self.camera_index = 0
        self.arduino_port = 'COM4'
        self.unlock_duration = 10
        
        self.encryption_key = self.generate_encryption_key()
        
    def generate_encryption_key(self):
        key_file = Path("encryption_key.key")
        if not key_file.exists():
            key = os.urandom(32)
            with open(key_file, 'wb') as f:
                f.write(key)
        else:
            with open(key_file, 'rb') as f:
                key = f.read()
        return key
    
    def numpy_to_serializable(self, obj):
        """Convert numpy arrays and other types to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'data': base64.b64encode(obj.tobytes()).decode('utf-8'),
                'shape': obj.shape
            }
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, (list, tuple)):
            return [self.numpy_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.numpy_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def serializable_to_numpy(self, obj):
        """Convert serializable format back to numpy arrays"""
        if isinstance(obj, dict) and '__numpy__' in obj:
            data = base64.b64decode(obj['data'])
            return np.frombuffer(data, dtype=obj['dtype']).reshape(obj['shape'])
        elif isinstance(obj, list):
            return [self.serializable_to_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.serializable_to_numpy(value) for key, value in obj.items()}
        else:
            return obj
    
    def simple_encrypt(self, data):
        """Encrypt data with XOR encryption"""
        # First convert all data to JSON serializable format
        serializable_data = self.numpy_to_serializable(data)
        
        # Convert to JSON and then to bytes
        try:
            json_data = json.dumps(serializable_data, ensure_ascii=False).encode('utf-8')
        except Exception as e:
            print(f"[JSON ERROR] Failed to serialize: {e}")
            # Fallback: convert numpy arrays to lists
            if isinstance(data, dict):
                safe_data = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        safe_data[key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64)):
                        safe_data[key] = float(value)
                    else:
                        safe_data[key] = value
                json_data = json.dumps(safe_data, ensure_ascii=False).encode('utf-8')
            else:
                raise e
        
        # XOR encryption
        key_extended = self.encryption_key * (len(json_data) // len(self.encryption_key) + 1)
        encrypted = bytes([json_data[i] ^ key_extended[i] for i in range(len(json_data))])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def simple_decrypt(self, encrypted_data):
        """Decrypt XOR encrypted data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            key_extended = self.encryption_key * (len(encrypted_bytes) // len(self.encryption_key) + 1)
            decrypted = bytes([encrypted_bytes[i] ^ key_extended[i] for i in range(len(encrypted_bytes))])
            
            serializable_data = json.loads(decrypted.decode('utf-8'))
            return self.serializable_to_numpy(serializable_data)
            
        except Exception as e:
            print(f"[DECRYPT ERROR] {e}")
            return None
    
    def detect_facial_landmarks(self, face_roi):
        """Detect facial landmarks (eyes, mouth, etc.)"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        landmarks = {
            'eyes': [],
            'mouth': [],
            'face_shape': [],
            'random_points': []
        }
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            landmarks['eyes'].append((int(center_x), int(center_y)))
            
            # Add random points around eyes
            for _ in range(3):
                rx = center_x + np.random.randint(-10, 10)
                ry = center_y + np.random.randint(-5, 5)
                landmarks['random_points'].append((int(rx), int(ry)))
        
        # Detect mouth
        mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.5, 15)
        for (mx, my, mw, mh) in mouths:
            center_x = mx + mw // 2
            center_y = my + mh // 2
            landmarks['mouth'].append((int(center_x), int(center_y)))
            
            # Add random points around mouth
            for _ in range(3):
                rx = center_x + np.random.randint(-15, 15)
                ry = center_y + np.random.randint(-5, 5)
                landmarks['random_points'].append((int(rx), int(ry)))
        
        # Face shape points
        h, w = gray_face.shape
        face_points = [
            (10, 10), (w-10, 10), (w//2, h-10),
            (w//4, h//3), (3*w//4, h//3),
            (w//4, 2*h//3), (3*w//4, 2*h//3)
        ]
        landmarks['face_shape'] = [(int(x), int(y)) for x, y in face_points]
        
        # Additional random points
        for _ in range(10):
            rx = np.random.randint(5, w-5)
            ry = np.random.randint(5, h-5)
            landmarks['random_points'].append((int(rx), int(ry)))
        
        return landmarks
    
    def create_face_signature(self, face_roi, landmarks):
        """Create face signature from landmarks"""
        all_points = []
        for category in ['eyes', 'mouth', 'face_shape', 'random_points']:
            all_points.extend(landmarks[category])
        
        points_array = np.array(all_points, dtype=np.float32)
        
        # Normalize coordinates
        if len(points_array) > 0:
            points_array = points_array / np.array([face_roi.shape[1], face_roi.shape[0]])
        
        # Add noise for security
        noise = np.random.normal(0, 0.01, points_array.shape)
        points_array += noise
        
        signature = points_array.flatten()
        if len(signature) < 128:
            signature = np.pad(signature, (0, 128 - len(signature)))
        
        return signature[:128]
    
    def build_cnn_model(self):
        """Build CNN model for face recognition"""
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='cosine_similarity')
        return model

    def initialize_system(self):
        """Initialize the complete system"""
        self.show_header()
        self.select_camera()
        self.setup_arduino()
        
        if not self.cnn_model:
            self.cnn_model = self.build_cnn_model()
            
        self.load_database()
        self.load_lbph_model()
        print("\n[SUCCESS] Professional FaceGate System Initialized!")
        
    def show_header(self):
        """Display system header"""
        print("=" * 70)
        print("           PROFESSIONAL FACEGATE SECURITY SYSTEM")
        print("=" * 70)
        print("Secure encrypted facial landmark recognition")
        print("Randomized point storage for maximum security")
        print("=" * 70)
        
    def select_camera(self):
        """Select camera from available options"""
        print("\n[CAMERA SELECTION]")
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"{i} - Camera {i} (Available)")
                cap.release()
        
        if not available_cameras:
            print("[ERROR] No cameras detected!")
            return
            
        while True:
            choice = input(f"Select camera ({', '.join(map(str, available_cameras))}): ").strip()
            if choice.isdigit() and int(choice) in available_cameras:
                self.camera_index = int(choice)
                print(f"[SUCCESS] Camera {self.camera_index} selected")
                break
            else:
                print("[ERROR] Invalid selection")
                
    def setup_arduino(self):
        """Setup Arduino connection"""
        print("\n[ARDUINO SETUP]")
        print("Available ports: COM1, COM2, COM3, COM4, COM5, COM6")
        port = input("Enter Arduino port (press Enter for COM4): ").strip()
        if port:
            self.arduino_port = port
            
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            time.sleep(2)
            
            while self.arduino.in_waiting > 0:
                self.arduino.readline()
                
            print(f"[SUCCESS] Arduino connected on {self.arduino_port}")
            
        except Exception as e:
            print(f"[WARNING] Arduino not found: {e}")
            print("[INFO] Running in simulation mode")
            self.arduino = None

    def load_database(self):
        """Load encrypted face database"""
        try:
            with open('secure_facegate_database.pkl', 'rb') as f:
                encrypted_db = pickle.load(f)
            
            self.face_db = {}
            for name, encrypted_data in encrypted_db.items():
                decrypted_data = self.simple_decrypt(encrypted_data)
                if decrypted_data is not None:
                    self.face_db[name] = decrypted_data
            
            print(f"[SUCCESS] Encrypted database loaded: {len(self.face_db)} persons")
        except Exception as e:
            print(f"[INFO] No database found: {e}")
            self.face_db = {}
            
    def save_database(self):
        """Save face database with encryption"""
        encrypted_db = {}
        for name, data in self.face_db.items():
            encrypted_db[name] = self.simple_encrypt(data)
        
        with open('secure_facegate_database.pkl', 'wb') as f:
            pickle.dump(encrypted_db, f)
        
        self.train_lbph_model()
        print("[SUCCESS] All data encrypted and saved")
        
    def train_lbph_model(self):
        """Train LBPH model with registered faces"""
        if not self.face_db or self.lbph_recognizer is None:
            return
            
        faces = []
        labels = []
        current_label = 0
        
        for name in self.face_db.keys():
            face_dir = Path(f"./secure_faces/{name}")
            if face_dir.exists():
                for img_path in face_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        faces.append(img)
                        labels.append(current_label)
            current_label += 1
        
        if faces and len(set(labels)) > 0:
            self.lbph_recognizer.train(faces, np.array(labels))
            self.lbph_recognizer.write('secure_facegate_lbph_model.xml')
            self.lbph_trained = True
            print(f"[SUCCESS] LBPH model trained with {len(faces)} images")

    def register_face(self):
        """Register new face with encryption"""
        print("\n[FACE REGISTRATION]")
        name = input("Enter person name: ").strip()
        if not name:
            print("[ERROR] Invalid name")
            return
            
        print(f"\nStarting encrypted registration for: {name}")
        print("Facial landmarks will be stored in encrypted format")
        
        face_dir = Path(f"./secure_faces/{name}")
        face_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.camera_index)
        cnn_samples = []
        landmark_signatures = []
        total_samples = 50
        
        print("\n[LANDMARK DETECTION ACTIVE]")
        print("Detecting eyes, mouth and facial features...")
        print("Generating randomized security points...")
        
        collected = 0
        while collected < total_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            face, bbox, analysis, landmarks = self.detect_and_analyze_face(frame)
            display_frame = frame.copy()
            
            if face is not None and analysis['quality'] == "HIGH":
                signature = self.create_face_signature(face, landmarks)
                landmark_signatures.append(signature)
                cnn_samples.append(face)
                
                x, y, w, h = bbox
                gray_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                cv2.imwrite(str(face_dir / f"{collected}.jpg"), gray_face)
                
                collected += 1
                
                self.draw_landmarks(display_frame, bbox, landmarks)
                cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,0), 3)
                cv2.putText(display_frame, "ENCRYPTING LANDMARKS...", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display_frame, f"Progress: {collected}/{total_samples}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            self.draw_registration_ui(display_frame, name, collected, total_samples)
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if len(cnn_samples) >= 30:
            print(f"\n[ENCRYPTION] Generating encrypted embeddings...")
            
            cnn_embeddings = []
            for sample in cnn_samples:
                embedding = self.cnn_model.predict(np.array([sample]), verbose=0)[0]
                cnn_embeddings.append(embedding)
            
            # Convert to Python native types for JSON serialization
            cnn_embeddings_serializable = [emb.tolist() for emb in cnn_embeddings]
            landmark_signatures_serializable = [sig.tolist() for sig in landmark_signatures]
            
            combined_data = {
                'cnn_embeddings': cnn_embeddings_serializable,
                'landmark_signatures': landmark_signatures_serializable,
                'registration_time': float(time.time()),
                'sample_count': int(len(cnn_samples))
            }
            
            self.face_db[name] = combined_data
            self.save_database()
            
            print(f"[SUCCESS] {name} securely registered!")
            print(f"Samples: {len(cnn_samples)}")
            print(f"Landmark signatures: {len(landmark_signatures)}")
            print(f"All data stored with XOR encryption")
            
            # Test recognition immediately
            self.test_recognition_after_registration(name)
        else:
            print("[ERROR] Not enough high-quality samples collected")
    
    def test_recognition_after_registration(self, name):
        """Test recognition immediately after registration"""
        print("\n[TESTING RECOGNITION]")
        print("Please look at the camera for recognition test...")
        
        cap = cv2.VideoCapture(self.camera_index)
        test_start = time.time()
        test_complete = False
        
        while time.time() - test_start < 10 and not test_complete:
            ret, frame = cap.read()
            if not ret:
                continue
                
            face, bbox, analysis, landmarks = self.detect_and_analyze_face(frame)
            if face is not None:
                test_name, confidence, method = self.secure_face_recognition(face, landmarks, frame)
                
                if test_name == name and confidence > self.threshold:
                    print(f"[TEST SUCCESS] Recognition working! Confidence: {confidence:.3f}")
                    test_complete = True
                elif time.time() - test_start > 3:
                    print(f"[TEST INFO] Current: {test_name}, Confidence: {confidence:.3f}")
            
            cv2.waitKey(1)
            
        cap.release()
        cv2.destroyAllWindows()
            
    def detect_and_analyze_face(self, frame):
        """Detect and analyze face in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(120,120)
        )
        
        if len(faces) == 0:
            return None, None, {}, {}
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = frame[y:y+h, x:x+w]
        
        landmarks = self.detect_facial_landmarks(face_roi)
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        
        if brightness > 80 and contrast > 40 and sharpness > 50 and len(eyes) >= 1:
            quality = "HIGH"
        else:
            quality = "LOW"
        
        analysis = {
            'quality': quality,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'eyes_detected': len(eyes)
        }
        
        face_processed = cv2.resize(face_roi, (128,128))
        face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        face_processed = face_processed.astype('float32') / 255.0
        
        return face_processed, (x,y,w,h), analysis, landmarks
    
    def draw_landmarks(self, frame, bbox, landmarks):
        """Draw facial landmarks on frame"""
        x, y, w, h = bbox
        
        for px, py in landmarks['eyes']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 3, (0, 255, 0), -1)
        
        for px, py in landmarks['mouth']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 3, (255, 0, 0), -1)
        
        for px, py in landmarks['face_shape']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 2, (0, 0, 255), -1)
        
        for px, py in landmarks['random_points']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 1, (0, 255, 255), -1)
    
    def draw_registration_ui(self, frame, name, current, total):
        """Draw registration UI"""
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0,0), (w,100), (0,0,0), -1)
        cv2.putText(frame, f"REGISTRATION: {name}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"ENCRYPTING FACIAL LANDMARKS...", (20,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"Progress: {current}/{total}", (20,85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        progress = int((current/total) * (w-40))
        cv2.rectangle(frame, (20, h-40), (w-20, h-20), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-40), (20+progress, h-20), (0,255,0), -1)
        
        cv2.putText(frame, "XOR Encrypted Landmark Storage", 
                   (w-400, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    def security_system(self):
        """Main security system loop"""
        print("\n[SECURITY SYSTEM ACTIVATED]")
        print("Press 'q' to exit, 'l' to manually lock")
        
        cap = cv2.VideoCapture(self.camera_index)
        door_unlocked = False
        unlock_time = 0
        last_person = "UNKNOWN"
        unknown_counter = 0
        is_scanning = False
        scanning_start_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            if door_unlocked and time.time() - unlock_time > self.unlock_duration:
                self.lock_door()
                door_unlocked = False
                print("[AUTO LOCK] Door locked")
                unknown_counter = 0
                is_scanning = False
                
            face, bbox, analysis, landmarks = self.detect_and_analyze_face(frame)
            
            if face is not None and not door_unlocked and not is_scanning:
                is_scanning = True
                scanning_start_time = time.time()
                print("[SCANNING] Encrypted face recognition started...")
            
            if is_scanning and face is not None:
                name, confidence, method = self.secure_face_recognition(face, landmarks, frame)
                
                if (name != "UNKNOWN" and name != "SCANNING..." and 
                    confidence > self.threshold and not door_unlocked):
                    
                    print(f"[ACCESS GRANTED] {name} - Confidence: {confidence:.3f}")
                    print(f"Encrypted landmark match: {method}")
                    if self.unlock_door():
                        door_unlocked = True
                        unlock_time = time.time()
                        last_person = name
                        unknown_counter = 0
                        is_scanning = False
                
                elif time.time() - scanning_start_time > 5:
                    print("[SCANNING TIMEOUT] No encrypted match found")
                    is_scanning = False
                    unknown_counter += 1
            
            if is_scanning:
                display_name = "SCANNING..."
                display_confidence = 0.0
                display_method = "ENCRYPTED_MATCH"
            else:
                if face is not None:
                    name, confidence, method = self.secure_face_recognition(face, landmarks, frame)
                    display_name = name
                    display_confidence = confidence
                    display_method = method
                else:
                    display_name = "NO FACE"
                    display_confidence = 0.0
                    display_method = "NO_FACE"
            
            self.draw_security_ui(frame, display_name, display_confidence, 
                               bbox, door_unlocked, last_person, 
                               display_method, unlock_time, is_scanning, landmarks)
            cv2.imshow('FaceGate Security', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l') and door_unlocked:
                self.lock_door()
                door_unlocked = False
                print("[MANUAL LOCK] Door locked")
                
        if door_unlocked:
            self.lock_door()
        cap.release()
        cv2.destroyAllWindows()
    
    def secure_face_recognition(self, face_processed, landmarks, frame):
        """Perform secure face recognition"""
        if not self.face_db:
            return "NO_DATA", 0.0, "NO_DATA"
        
        current_signature = self.create_face_signature(face_processed, landmarks)
        query_embedding = self.cnn_model.predict(np.array([face_processed]), verbose=0)[0]
        
        best_match = "UNKNOWN"
        best_similarity = 0.0
        best_method = "NO_MATCH"
        
        for name, stored_data in self.face_db.items():
            cnn_similarity = 0.0
            # Convert stored embeddings back to numpy arrays
            stored_embeddings = [np.array(emb, dtype=np.float32) for emb in stored_data['cnn_embeddings']]
            
            for emb in stored_embeddings:
                similarity = self.cosine_similarity(query_embedding, emb)
                if similarity > cnn_similarity:
                    cnn_similarity = similarity
            
            landmark_similarity = 0.0
            # Convert stored signatures back to numpy arrays
            stored_signatures = [np.array(sig, dtype=np.float32) for sig in stored_data['landmark_signatures']]
            
            for stored_sig in stored_signatures:
                similarity = self.cosine_similarity(current_signature, stored_sig)
                if similarity > landmark_similarity:
                    landmark_similarity = similarity
            
            combined_similarity = (cnn_similarity + landmark_similarity) / 2
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = name
                best_method = "ENCRYPTED_LANDMARKS" if landmark_similarity > cnn_similarity else "CNN_EMBEDDING"
        
        return best_match, best_similarity, best_method
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)
        
    def unlock_door(self):
        """Unlock the door"""
        if self.arduino:
            try:
                self.arduino.write(b"UNLOCK\n")
                time.sleep(0.5)
                print("[ARDUINO] Door unlocked")
                return True
            except Exception as e:
                print(f"[ERROR] Arduino communication: {e}")
                return False
        else:
            print("[SIMULATION] Door unlocked")
            return True
            
    def lock_door(self):
        """Lock the door"""
        if self.arduino:
            try:
                self.arduino.write(b"LOCK\n")
                time.sleep(0.5)
                print("[ARDUINO] Door locked")
                return True
            except Exception as e:
                print(f"[ERROR] Arduino communication: {e}")
                return False
        else:
            print("[SIMULATION] Door locked")
            return True
            
    def draw_security_ui(self, frame, name, confidence, bbox, unlocked, last_person, method, unlock_time, is_scanning, landmarks=None):
        """Draw security system UI"""
        h, w = frame.shape[:2]
        
        if is_scanning:
            bg_color = (0, 50, 100)
        elif name == "UNKNOWN":
            bg_color = (0, 0, 100)
        else:
            bg_color = (0, 0, 0)
            
        cv2.rectangle(frame, (0,0), (w,140), bg_color, -1)
        
        status_color = (0,255,0) if unlocked else (0,0,255)
        status_text = "UNLOCKED" if unlocked else "SCANNING..." if is_scanning else "LOCKED"
        cv2.putText(frame, f"STATUS: {status_text}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if is_scanning:
            name_color = (0,255,255)
            name_text = "ENCRYPTED SCANNING..."
        else:
            name_color = (0,255,0) if name not in ["UNKNOWN", "NO FACE", "NO_DATA"] else (0,0,255)
            name_text = name
            
        cv2.putText(frame, f"IDENTITY: {name_text}", (20,65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 2)
        
        if not is_scanning and name not in ["NO FACE", "SCANNING..."]:
            cv2.putText(frame, f"CONFIDENCE: {confidence:.3f}", (20,90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"METHOD: {method}", (20,110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        if bbox and landmarks:
            self.draw_landmarks(frame, bbox, landmarks)
            x, y, w_rect, h_rect = bbox
            box_color = (0,255,255) if is_scanning else (0,255,0) if name not in ["UNKNOWN", "NO FACE"] else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w_rect,y+h_rect), box_color, 3)
            
            if is_scanning:
                for i in range(8):
                    offset = (int(time.time() * 10) + i) % 30
                    cv2.circle(frame, (x + offset, y + 10), 5, (0,255,255), -1)
        
        cv2.rectangle(frame, (0,h-40), (w,h), (0,0,0), -1)
        cv2.putText(frame, f"Last: {last_person}", (20, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, "Q: Exit | L: Lock", (w-150, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, "Encrypted Landmarks Active", (w-300, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        if unlocked:
            elapsed = time.time() - unlock_time
            remaining = max(0, self.unlock_duration - elapsed)
            cv2.putText(frame, f"AUTO LOCK: {remaining:.1f}s", (w-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
    def main_menu(self):
        """Main menu system"""
        self.initialize_system()
        
        while True:
            print("\n" + "=" * 50)
            print("          PROFESSIONAL FACEGATE SYSTEM")
            print("=" * 50)
            print("1. Register New Face (Encrypted)")
            print("2. Start Security System") 
            print("3. System Status")
            print("4. Exit")
            print("-" * 50)
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                self.register_face()
            elif choice == '2':
                self.security_system()
            elif choice == '3':
                self.show_system_status()
            elif choice == '4':
                print("\n[SYSTEM SHUTDOWN]")
                if self.arduino:
                    self.arduino.close()
                break
            else:
                print("[ERROR] Invalid selection")
                
    def show_system_status(self):
        """Display system status"""
        print("\n[SYSTEM STATUS]")
        print(f"Camera: {self.camera_index}")
        print(f"Arduino: {'Connected' if self.arduino else 'Disconnected'}")
        print(f"Registered Persons: {len(self.face_db)}")
        print(f"Encryption: XOR Active")
        print(f"Recognition: CNN + Encrypted Landmarks")
        print(f"Security Threshold: {self.threshold}")
        print("All facial data stored with randomized encrypted landmarks")

    def load_lbph_model(self):
        """Load LBPH model"""
        if self.lbph_recognizer is None:
            return
            
        try:
            self.lbph_recognizer.read('secure_facegate_lbph_model.xml')
            self.lbph_trained = True
            print("[SUCCESS] LBPH model loaded")
        except:
            print("[INFO] No LBPH model found")
            self.lbph_trained = False

if __name__ == "__main__":
    system = ProfessionalFaceGate()
    system.main_menu()