import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import serial
import pickle
import os

class ProfessionalFaceLockSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.face_db = {}
        self.threshold = 0.75
        self.arduino = None
        self.camera_index = 0
        self.arduino_port = 'COM3'
        self.unlock_duration = 10
        
    def initialize_system(self):
        self.show_header()
        self.select_camera()
        self.setup_arduino()
        
        if not self.model:
            self.model = self.build_model()
            
        self.load_database()
        print("\n[SUCCESS] System initialized!")
        
    def show_header(self):
        print("=" * 70)
        print("           PROFESSIONAL AI FACE RECOGNITION LOCK SYSTEM")
        print("=" * 70)
        print()
        
    def select_camera(self):
        print("[CAMERA SELECTION]")
        print("0 - Primary Camera")
        print("1 - Secondary Camera") 
        print("2 - External Camera")
        
        while True:
            choice = input("Select camera (0-2): ").strip()
            if choice in ['0', '1', '2']:
                self.camera_index = int(choice)
                
                # Test camera
                cap = cv2.VideoCapture(self.camera_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        print(f"[SUCCESS] Camera {self.camera_index} selected")
                        break
                    else:
                        print("[ERROR] Camera test failed")
                else:
                    print("[ERROR] Camera not accessible")
            else:
                print("[ERROR] Invalid selection")
                
    def setup_arduino(self):
        print("\n[ARDUINO SETUP]")
        print("Available ports: COM3, COM4, COM5, COM6, /dev/ttyUSB0, /dev/ttyACM0")
        port = input("Enter Arduino port (press Enter for COM3): ").strip()
        if port:
            self.arduino_port = port
            
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            time.sleep(2)
            
            # Clear buffer
            while self.arduino.in_waiting > 0:
                self.arduino.readline()
                
            print(f"[SUCCESS] Arduino connected on {self.arduino_port}")
            
        except Exception as e:
            print(f"[WARNING] Arduino not found: {e}")
            print("[INFO] Running in simulation mode")
            self.arduino = None
            
    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def load_database(self):
        try:
            with open('face_database.pkl', 'rb') as f:
                self.face_db = pickle.load(f)
            print(f"[SUCCESS] Database loaded: {len(self.face_db)} persons")
        except:
            print("[INFO] No database found - starting fresh")
            
    def save_database(self):
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(self.face_db, f)
        print("[SUCCESS] Database saved")
        
    def register_face(self):
        print("\n[FACE REGISTRATION]")
        name = input("Enter person name: ").strip()
        if not name:
            print("[ERROR] Invalid name")
            return
            
        print(f"Starting registration for: {name}")
        print("Look directly at the camera")
        print("Press 'q' to finish, 'r' to retry")
        
        cap = cv2.VideoCapture(self.camera_index)
        samples = []
        count = 0
        target_samples = 25
        
        while count < target_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            face, bbox, analysis = self.detect_and_analyze_face(frame)
            display_frame = frame.copy()
            
            if face is not None:
                samples.append(face)
                count += 1
                
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,0), 3)
                
                # Visual analysis
                cv2.putText(display_frame, "SCANNING FACE", (x, y-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(display_frame, f"Quality: {analysis['quality']}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            # UI
            self.draw_registration_ui(display_frame, name, count, target_samples)
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                samples = []
                count = 0
                
        cap.release()
        cv2.destroyAllWindows()
        
        if samples:
            embeddings = []
            for sample in samples:
                embedding = self.model.predict(np.array([sample]), verbose=0)[0]
                embeddings.append(embedding)
                
            self.face_db[name] = embeddings
            self.save_database()
            print(f"[SUCCESS] {name} registered with {len(samples)} samples")
        else:
            print("[ERROR] No samples collected")
            
    def detect_and_analyze_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120,120))
        
        if len(faces) == 0:
            return None, None, {}
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Face analysis
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        
        quality = "HIGH" if brightness > 100 and contrast > 50 else "LOW"
        
        analysis = {
            'quality': quality,
            'brightness': brightness,
            'contrast': contrast
        }
        
        # Process face
        face_processed = cv2.resize(face_roi, (128,128))
        face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        face_processed = face_processed.astype('float32') / 255.0
        
        return face_processed, (x,y,w,h), analysis
        
    def draw_registration_ui(self, frame, name, count, target):
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0,0), (w,80), (0,0,0), -1)
        cv2.putText(frame, f"REGISTRATION: {name}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"SAMPLES: {count}/{target}", (20,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Progress bar
        progress = int((count/target) * (w-40))
        cv2.rectangle(frame, (20, h-50), (w-20, h-30), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-50), (20+progress, h-30), (0,255,0), -1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to finish, 'r' to restart", 
                   (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                   
    def security_system(self):
        print("\n[SECURITY SYSTEM ACTIVATED]")
        print("Press 'q' to exit, 'l' to manually lock")
        
        cap = cv2.VideoCapture(self.camera_index)
        door_unlocked = False
        unlock_time = 0
        last_person = "UNKNOWN"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Auto lock after duration
            if door_unlocked and time.time() - unlock_time > self.unlock_duration:
                self.lock_door()
                door_unlocked = False
                print("[AUTO LOCK] Door locked")
                
            # Face recognition
            name, confidence, bbox, analysis = self.recognize_face(frame)
            
            # Security decision
            if (name != "UNKNOWN" and confidence > self.threshold and 
                not door_unlocked and name != "NO_FACE"):
                
                if self.unlock_door():
                    door_unlocked = True
                    unlock_time = time.time()
                    last_person = name
                    print(f"[ACCESS GRANTED] {name} - Confidence: {confidence:.3f}")
                    
            elif name == "UNKNOWN" and bbox:
                print(f"[ACCESS DENIED] Unknown person detected")
                
            # Draw UI
            self.draw_security_ui(frame, name, confidence, bbox, door_unlocked, 
                                last_person, analysis)
            cv2.imshow('Security System', frame)
            
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
        
    def recognize_face(self, frame):
        if not self.face_db:
            return "NO_DATA", 0.0, None, {}
            
        face, bbox, analysis = self.detect_and_analyze_face(frame)
        if face is None:
            return "NO_FACE", 0.0, None, {}
            
        query_embedding = self.model.predict(np.array([face]), verbose=0)[0]
        
        best_match = "UNKNOWN"
        best_similarity = 0.0
        
        for name, embeddings in self.face_db.items():
            for emb in embeddings:
                similarity = self.cosine_similarity(query_embedding, emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
                    
        return best_match, best_similarity, bbox, analysis
        
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def unlock_door(self):
        if self.arduino:
            try:
                self.arduino.write(b"UNLOCK\n")
                time.sleep(0.5)
                return True
            except:
                print("[ERROR] Arduino communication failed")
                return False
        else:
            print("[SIMULATION] Door unlocked")
            return True
            
    def lock_door(self):
        if self.arduino:
            try:
                self.arduino.write(b"LOCK\n")
                time.sleep(0.5)
                return True
            except:
                print("[ERROR] Arduino communication failed")
                return False
        else:
            print("[SIMULATION] Door locked")
            return True
            
    def draw_security_ui(self, frame, name, confidence, bbox, unlocked, last_person, analysis):
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0,0), (w,100), (0,0,0), -1)
        
        # System status
        status_color = (0,255,0) if unlocked else (0,0,255)
        status_text = "UNLOCKED" if unlocked else "LOCKED"
        cv2.putText(frame, f"STATUS: {status_text}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Recognition info
        name_color = (0,255,0) if name not in ["UNKNOWN", "NO_FACE", "NO_DATA"] else (0,0,255)
        cv2.putText(frame, f"IDENTIFIED: {name}", (20,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 2)
        cv2.putText(frame, f"CONFIDENCE: {confidence:.3f}", (20,85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Face bounding box
        if bbox:
            x, y, w_rect, h_rect = bbox
            box_color = (0,255,0) if name not in ["UNKNOWN", "NO_FACE", "NO_DATA"] else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w_rect,y+h_rect), box_color, 3)
            
            # Scanning animation
            for i in range(3):
                offset = (int(time.time() * 5) + i) % 20
                cv2.circle(frame, (x + offset, y + 5), 3, (0,255,255), -1)
                
            # Analysis info
            cv2.putText(frame, f"Quality: {analysis.get('quality', 'N/A')}", 
                       (x, y+h_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Footer
        cv2.rectangle(frame, (0,h-40), (w,h), (0,0,0), -1)
        cv2.putText(frame, f"Last access: {last_person}", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, "Q: Exit | L: Manual Lock", (w-200, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Unlock timer
        if unlocked:
            elapsed = time.time() - unlock_time
            remaining = max(0, self.unlock_duration - elapsed)
            cv2.putText(frame, f"AUTO LOCK IN: {remaining:.1f}s", (w-300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
    def main_menu(self):
        self.initialize_system()
        
        while True:
            print("\n" + "=" * 50)
            print("          MAIN MENU")
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
        print(f"Registered persons: {len(self.face_db)}")
        print(f"Recognition threshold: {self.threshold}")
        print(f"Unlock duration: {self.unlock_duration}s")
        print(f"Model: {'Loaded' if self.model else 'Not loaded'}")

# Global variables
unlock_time = 0

if __name__ == "__main__":
    system = ProfessionalFaceLockSystem()
    system.main_menu()