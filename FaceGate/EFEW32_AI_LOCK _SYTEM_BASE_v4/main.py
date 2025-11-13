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

class ProfessionalFaceGate:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Kétféle modell - jobb biztonság érdekében
        self.cnn_model = None
        
        # LBPH recognizer - kompatibilis verzióval
        try:
            # OpenCV 4.x verzió
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                # OpenCV 3.x verzió
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                # Ha egyik sem működik, használjunk alternatívát
                print("[WARNING] LBPH not available, using CNN only")
                self.lbph_recognizer = None
        
        self.face_db = {}
        self.lbph_trained = False
        
        self.threshold = 0.75  # Realisztikus küszöb
        self.arduino = None
        self.camera_index = 0
        self.arduino_port = 'COM4'
        self.unlock_duration = 10
        
    def initialize_system(self):
        self.show_header()
        self.select_camera()
        self.setup_arduino()
        
        if not self.cnn_model:
            self.cnn_model = self.build_cnn_model()
            
        self.load_database()
        self.load_lbph_model()
        print("\n[SUCCESS] Professional FaceGate System Initialized!")
        
    def show_header(self):
        print("=" * 70)
        print("           PROFESSIONAL FACEGATE SECURITY SYSTEM")
        print("=" * 70)
        print()
        
    def select_camera(self):
        print("[CAMERA SELECTION]")
        
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
            
    def build_cnn_model(self):
        """Egyszerűbb CNN modell gyorsabb és pontosabb működésért"""
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
        
    def load_database(self):
        """CNN adatbázis betöltése"""
        try:
            with open('facegate_cnn_database.pkl', 'rb') as f:
                self.face_db = pickle.load(f)
            print(f"[SUCCESS] CNN database loaded: {len(self.face_db)} persons")
        except:
            print("[INFO] No CNN database found")
            self.face_db = {}
            
    def load_lbph_model(self):
        """LBPH modell betöltése"""
        if self.lbph_recognizer is None:
            return
            
        try:
            self.lbph_recognizer.read('facegate_lbph_model.xml')
            self.lbph_trained = True
            print("[SUCCESS] LBPH model loaded")
        except:
            print("[INFO] No LBPH model found - will train after registration")
            self.lbph_trained = False
            
    def save_database(self):
        """Minden adat mentése"""
        # CNN adatbázis
        with open('facegate_cnn_database.pkl', 'wb') as f:
            pickle.dump(self.face_db, f)
        
        # LBPH modell újratanítása
        self.train_lbph_model()
        
        print("[SUCCESS] All data saved")
        
    def train_lbph_model(self):
        """LBPH modell tanítása a regisztrált arcokról"""
        if not self.face_db or self.lbph_recognizer is None:
            return
            
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        # Adatok előkészítése LBPH-hoz
        for name, embeddings in self.face_db.items():
            # Mentsük el az arcokat LBPH számára
            face_dir = Path(f"./faces/{name}")
            if face_dir.exists():
                for img_path in face_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Átméretezés konzisztenciáért
                        img = cv2.resize(img, (100, 100))
                        faces.append(img)
                        labels.append(current_label)
                        
            label_dict[current_label] = name
            current_label += 1
        
        if faces and len(set(labels)) > 0:
            self.lbph_recognizer.train(faces, np.array(labels))
            self.lbph_recognizer.write('facegate_lbph_model.xml')
            self.lbph_trained = True
            print(f"[SUCCESS] LBPH model trained with {len(faces)} images")
            
    def register_face(self):
        print("\n[FACE REGISTRATION]")
        name = input("Enter person name: ").strip()
        if not name:
            print("[ERROR] Invalid name")
            return
            
        print(f"\nStarting advanced registration for: {name}")
        print("This will include multi-angle face capture for better accuracy")
        
        # Könyvtár létrehozása LBPH számára
        face_dir = Path(f"./faces/{name}")
        face_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.camera_index)
        cnn_samples = []
        lbph_count = 0
        total_samples = 60  # Több minta különböző szögekből
        
        # Regisztrációs lépések
        steps = [
            ("Look straight at the camera", 15),
            ("Slowly turn head LEFT", 15),
            ("Slowly turn head RIGHT", 15),
            ("Look slightly UP", 8),
            ("Look slightly DOWN", 7)
        ]
        
        current_step = 0
        step_samples = 0
        
        for step_name, step_target in steps:
            print(f"\n[STEP {current_step + 1}/5] {step_name}")
            print(f"Capture {step_target} samples...")
            
            step_samples = 0
            while step_samples < step_target:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                face, bbox, analysis = self.detect_and_analyze_face(frame)
                display_frame = frame.copy()
                
                if face is not None and analysis['quality'] == "HIGH":
                    cnn_samples.append(face)
                    
                    # LBPH számára is mentjük a képeket
                    x, y, w, h = bbox
                    gray_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (100, 100))  # Standard méret LBPH-hoz
                    cv2.imwrite(str(face_dir / f"{lbph_count}.jpg"), gray_face)
                    lbph_count += 1
                    step_samples += 1
                    
                    # UI
                    cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,0), 3)
                    cv2.putText(display_frame, f"STEP {current_step + 1}: {step_name}", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(display_frame, f"Progress: {step_samples}/{step_target}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.putText(display_frame, f"Quality: {analysis['quality']}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                
                self.draw_registration_ui(display_frame, name, len(cnn_samples), total_samples, step_name)
                cv2.imshow('Face Registration', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            if key == ord('q'):
                break
                
            current_step += 1
                
        cap.release()
        cv2.destroyAllWindows()
        
        if len(cnn_samples) >= 30:
            print(f"[PROCESSING] Generating embeddings from {len(cnn_samples)} samples...")
            
            embeddings = []
            for sample in cnn_samples:
                embedding = self.cnn_model.predict(np.array([sample]), verbose=0)[0]
                embeddings.append(embedding)
                
            self.face_db[name] = embeddings
            self.save_database()
            print(f"[SUCCESS] {name} registered with {len(cnn_samples)} samples from multiple angles")
        else:
            print("[ERROR] Not enough high-quality samples collected")
            
    def draw_registration_ui(self, frame, name, current, total, instruction):
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0,0), (w,100), (0,0,0), -1)
        cv2.putText(frame, f"REGISTERING: {name}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"TOTAL PROGRESS: {current}/{total}", (20,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"INSTRUCTION: {instruction}", (20,85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        progress = int((current/total) * (w-40))
        cv2.rectangle(frame, (20, h-40), (w-20, h-20), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-40), (20+progress, h-20), (0,255,0), -1)
        
    def detect_and_analyze_face(self, frame):
        """Arcészlelés és minőségellenőrzés"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(120,120)
        )
        
        if len(faces) == 0:
            return None, None, {}
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Szemérzékelés
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        # Minőségellenőrzés
        face_color = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        if brightness > 80 and contrast > 40 and sharpness > 50 and len(eyes) >= 1:
            quality = "HIGH"
        else:
            quality = "LOW"
        
        analysis = {
            'quality': quality,
            'brightness': brightness,
            'contrast': contrast,
            'eyes_detected': len(eyes)
        }
        
        # Arc előfeldolgozás
        face_processed = cv2.resize(face_color, (128,128))
        face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        face_processed = face_processed.astype('float32') / 255.0
        
        return face_processed, (x,y,w,h), analysis
        
    def security_system(self):
        print("\n[SECURITY SYSTEM ACTIVATED]")
        print("Press 'q' to exit, 'l' to manually lock")
        
        cap = cv2.VideoCapture(self.camera_index)
        door_unlocked = False
        unlock_time = 0
        last_person = "UNKNOWN"
        unknown_counter = 0
        recognition_loader = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Automatikus zárás
            if door_unlocked and time.time() - unlock_time > self.unlock_duration:
                self.lock_door()
                door_unlocked = False
                print("[AUTO LOCK] Door locked")
                unknown_counter = 0
                
            # KETTŐS ARCFELISMERÉS - CNN + LBPH
            name, confidence, bbox, method = self.dual_face_recognition(frame)
            
            # Loader animáció
            recognition_loader = (recognition_loader + 1) % 100
            
            # BIZTONSÁGI DÖNTÉS
            if (name != "UNKNOWN" and name != "NO_FACE" and 
                confidence > self.threshold and not door_unlocked):
                
                if self.verify_identity(name, confidence):
                    if self.unlock_door():
                        door_unlocked = True
                        unlock_time = time.time()
                        last_person = name
                        unknown_counter = 0
                        print(f"[ACCESS GRANTED] {name} - Confidence: {confidence:.3f}")
                        
            # ISMERETLEN ARC
            elif name == "UNKNOWN" and bbox:
                unknown_counter += 1
                if unknown_counter >= 5:  # 5 ismeretlen észlelés után riasztás
                    print(f"[SECURITY ALERT] Unknown person detected!")
                    self.lock_door()
                    door_unlocked = False
                    
            elif name == "NO_FACE":
                unknown_counter = 0
                
            # UI
            self.draw_security_ui(frame, name, confidence, bbox, door_unlocked, 
                                last_person, method, unlock_time, recognition_loader)
            cv2.imshow('FaceGate Security', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l') and door_unlocked:
                self.lock_door()
                door_unlocked = False
                unknown_counter = 0
                print("[MANUAL LOCK] Door locked")
                
        if door_unlocked:
            self.lock_door()
        cap.release()
        cv2.destroyAllWindows()
        
    def dual_face_recognition(self, frame):
        """Kettős arcfelismerés - CNN + LBPH"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120,120))
        
        if len(faces) == 0:
            return "NO_FACE", 0.0, None, "NONE"
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        bbox = (x, y, w, h)
        
        # 1. LBPH felismerés (pontosabb ismeretlen detektálás)
        lbph_confidence = 0.0
        if self.lbph_trained and self.lbph_recognizer is not None:
            try:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (100, 100))  # Standard méret
                label, confidence = self.lbph_recognizer.predict(roi_gray)
                lbph_confidence = (100 - confidence) / 100  # Normalizálás 0-1 közé
            except Exception as e:
                print(f"[LBPH ERROR] {e}")
        
        # 2. CNN felismerés
        cnn_name, cnn_confidence = "UNKNOWN", 0.0
        face_color = frame[y:y+h, x:x+w]
        face_processed = cv2.resize(face_color, (128,128))
        face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        face_processed = face_processed.astype('float32') / 255.0
        
        if self.face_db:
            query_embedding = self.cnn_model.predict(np.array([face_processed]), verbose=0)[0]
            
            best_match = "UNKNOWN"
            best_similarity = 0.0
            
            for name, embeddings in self.face_db.items():
                for emb in embeddings:
                    similarity = self.cosine_similarity(query_embedding, emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
            
            cnn_name = best_match
            cnn_confidence = best_similarity
        
        # DÖNTÉS - LBPH segíti az ismeretlen detektálást
        final_confidence = (cnn_confidence + lbph_confidence) / 2
        
        if lbph_confidence > 0.6 and cnn_confidence > self.threshold and cnn_name != "UNKNOWN":
            return cnn_name, final_confidence, bbox, "DUAL_VERIFIED"
        elif cnn_confidence > self.threshold and cnn_name != "UNKNOWN":
            return cnn_name, cnn_confidence, bbox, "CNN_ONLY"
        else:
            return "UNKNOWN", final_confidence, bbox, "REJECTED"
        
    def verify_identity(self, name, confidence):
        """Identitás ellenőrzése"""
        return confidence > self.threshold
        
    def cosine_similarity(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)
        
    def unlock_door(self):
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
            
    def draw_security_ui(self, frame, name, confidence, bbox, unlocked, last_person, method, unlock_time, loader):
        h, w = frame.shape[:2]
        
        # Háttér
        bg_color = (0, 0, 100) if name == "UNKNOWN" else (0, 0, 0)
        cv2.rectangle(frame, (0,0), (w,140), bg_color, -1)
        
        # Státusz
        status_color = (0,255,0) if unlocked else (0,0,255)
        status_text = "UNLOCKED" if unlocked else "LOCKED"
        cv2.putText(frame, f"STATUS: {status_text}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Felismerés
        name_color = (0,255,0) if name not in ["UNKNOWN", "NO_FACE"] else (0,0,255)
        cv2.putText(frame, f"IDENTITY: {name}", (20,65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 2)
        cv2.putText(frame, f"CONFIDENCE: {confidence:.3f}", (20,90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"METHOD: {method}", (20,110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Loader animáció
        loader_text = "SCANNING" + "." * ((loader // 20) % 4)
        cv2.putText(frame, loader_text, (w-150, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        # Arc keret
        if bbox:
            x, y, w_rect, h_rect = bbox
            box_color = (0,255,0) if name not in ["UNKNOWN", "NO_FACE"] else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w_rect,y+h_rect), box_color, 3)
            
            # Animáció
            for i in range(6):
                offset = (int(time.time() * 8) + i) % 30
                cv2.circle(frame, (x + offset, y + 10), 5, (0,255,255), -1)
        
        # Lábléc
        cv2.rectangle(frame, (0,h-40), (w,h), (0,0,0), -1)
        cv2.putText(frame, f"Last: {last_person}", (20, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, "Q: Exit | L: Lock", (w-150, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Visszaszámlálás
        if unlocked:
            elapsed = time.time() - unlock_time
            remaining = max(0, self.unlock_duration - elapsed)
            cv2.putText(frame, f"AUTO LOCK: {remaining:.1f}s", (w-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
    def main_menu(self):
        self.initialize_system()
        
        while True:
            print("\n" + "=" * 50)
            print("          PROFESSIONAL FACEGATE")
            print("=" * 50)
            print("1. Register New Face")
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
        print("\n[SYSTEM STATUS]")
        print(f"Camera: {self.camera_index}")
        print(f"Arduino: {'Connected' if self.arduino else 'Disconnected'}")
        print(f"Arduino Port: {self.arduino_port}")
        print(f"Registered Persons: {len(self.face_db)}")
        print(f"LBPH Trained: {self.lbph_trained}")
        print(f"Security Threshold: {self.threshold}")
        print(f"Recognition Method: Dual (CNN + LBPH)")

if __name__ == "__main__":
    system = ProfessionalFaceGate()
    system.main_menu()