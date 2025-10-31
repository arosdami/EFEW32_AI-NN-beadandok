import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== DEEP LEARNING ARCFELISMERŐ RENDSZER ===")
print("SIAMESE NEURÁLIS HÁLÓ - BEADANDÓ PROJEKT")
print("TensorFlow verzió:", tf.__version__)

class FaceDataGenerator:
    """
    Arc adat generátor és előfeldolgozó osztály
    """
    
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_and_align_face(self, image):
        """
        Arc detektálás és igazítás deep learning módszerekkel
        """
        if image is None:
            return None, None
            
        try:
            # Haar cascade arc detektálás
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Legnagyobb arc kiválasztása
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # Arc kivágása
                face_roi = image[y:y+h, x:x+w]
                
                # Méret normalizálás
                face_roi = cv2.resize(face_roi, self.target_size)
                
                # Előfeldolgozás
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_roi = face_roi.astype('float32')
                face_roi = face_roi / 255.0  # Normalizálás
                
                return face_roi, (x, y, w, h)
            
            return None, None
            
        except Exception as e:
            print(f"Hiba az arc detektálásnál: {e}")
            return None, None
    
    def collect_face_data(self, person_name, num_samples=200):
        """
        Arc adatok gyűjtése egy személyről
        """
        cap = cv2.VideoCapture(1)  # A te kamerád indexe
        
        if not cap.isOpened():
            print("Hiba: Kamera nem elérhető!")
            return []
        
        print(f"\nAdatgyűjtés indul: {person_name}")
        print(f"Minták száma: {num_samples}")
        print("Nézz közvetlenül a kamerába és változtasd a fej helyzetét!")
        print("Nyomj 'q'-t a kilépéshez")
        
        faces = []
        count = 0
        failed_attempts = 0
        
        while count < num_samples and failed_attempts < 50:
            ret, frame = cap.read()
            if not ret:
                failed_attempts += 1
                continue
            
            # Display frame inicializálása
            display_frame = frame.copy()
            
            # Arc detektálás és igazítás
            face, bbox = self.detect_and_align_face(frame)
            
            if face is not None:
                faces.append(face)
                count += 1
                failed_attempts = 0
                
                # Vizuális visszajelzés
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f'{person_name}: {count}/{num_samples}', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # HUD megjelenítés
            cv2.putText(display_frame, f"Minták: {count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Sikertelen próbálkozások: {failed_attempts}/50", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Nyomj 'q'-t a kilépéshez", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Adatgyűjtés - {person_name}', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"{len(faces)} arc minta gyűjtve {person_name} számára")
        return faces

class SiameseNetwork:
    """
    Siamese Neurális Háló implementáció arcfelismerésre
    Két azonos architektúrájú háló, amelyek közös súlyokat használnak
    """
    
    def __init__(self, input_shape=(160, 160, 3)):
        self.input_shape = input_shape
        self.model = None
        self.embedding_model = None
        
    def build_base_network(self):
        """
        Alap konvolúciós háló építése - közös része a Siamese hálónak
        """
        input = layers.Input(shape=self.input_shape)
        
        # Első konvolúciós blokk
        x = layers.Conv2D(64, (10, 10), activation='relu')(input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Második konvolúciós blokk
        x = layers.Conv2D(128, (7, 7), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Harmadik konvolúciós blokk
        x = layers.Conv2D(128, (4, 4), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Negyedik konvolúciós blokk
        x = layers.Conv2D(256, (4, 4), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Teljesen összekapcsolt rétegek
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='sigmoid')(x)
        
        return keras.Model(input, x)
    
    def build_siamese_network(self):
        """
        Siamese háló építése - két azonos alapháló
        """
        # Közös alap háló
        base_network = self.build_base_network()
        
        # Két bemeneti ág
        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)
        
        # Közös súlyokkal feldolgozás
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # L1 távolság a beágyazások között
        l1_distance = layers.Lambda(
            lambda tensors: tf.abs(tensors[0] - tensors[1])
        )([processed_a, processed_b])
        
        # Osztályozó réteg
        output = layers.Dense(1, activation='sigmoid')(l1_distance)
        
        # Siamese modell
        siamese_model = keras.Model([input_a, input_b], output)
        
        return siamese_model, base_network
    
    def compile_model(self, learning_rate=0.0001):
        """
        Modell kompilálása
        """
        self.model, self.embedding_model = self.build_siamese_network()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Siamese háló architektúra:")
        print(f"Bemeneti forma: {self.input_shape}")
        print(f"Beágyazás dimenzió: 4096")
    
    def create_pairs(self, faces, labels, num_pairs=1000):
        """
        Pozitív és negatív párok generálása a tanításhoz
        """
        # Egyszerűbb megközelítés - csak alap párokat generálunk
        pairs = []
        pair_labels = []
        
        # Pozitív párok (azonos személy)
        for i in range(len(faces)):
            for j in range(i+1, min(i+3, len(faces))):  # Korlátozzuk a párokat
                if labels[i] == labels[j]:
                    pairs.append([faces[i], faces[j]])
                    pair_labels.append(1)  # Pozitív pár
        
        # Negatív párok (különböző személyek)
        unique_labels = np.unique(labels)
        for i in range(len(pair_labels)):  # Ugyanannyi negatív pár
            # Véletlenszerűen választunk két különböző címkéjű arcot
            available_indices = list(range(len(faces)))
            idx1 = np.random.choice(available_indices)
            
            # Második index, ami más címkéjű
            available_indices = [idx for idx in available_indices if labels[idx] != labels[idx1]]
            if available_indices:
                idx2 = np.random.choice(available_indices)
                pairs.append([faces[idx1], faces[idx2]])
                pair_labels.append(0)  # Negatív pár
        
        return np.array(pairs), np.array(pair_labels)
    
    def train(self, faces, labels, epochs=20, batch_size=16):
        """
        Siamese háló tanítása
        """
        if len(faces) < 2:
            print("Hiba: Nincs elég adat a tanításhoz!")
            return None
        
        # Párok generálása
        pairs, pair_labels = self.create_pairs(faces, labels)
        
        if len(pairs) == 0:
            print("Hiba: Nem sikerült párokat generálni!")
            return None
        
        # Adatok felosztása
        X_train_pairs, X_test_pairs, y_train, y_test = train_test_split(
            pairs, pair_labels, test_size=0.2, random_state=42
        )
        
        # Bemenetek szétválasztása
        X_train = [X_train_pairs[:, 0], X_train_pairs[:, 1]]
        X_test = [X_test_pairs[:, 0], X_test_pairs[:, 1]]
        
        print(f"\nTanítási adatok:")
        print(f"  Párok száma: {len(pairs)}")
        print(f"  Pozitív párok: {np.sum(pair_labels)}")
        print(f"  Negatív párok: {len(pair_labels) - np.sum(pair_labels)}")
        
        # Callback-ek
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Tanítás
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def compute_embedding(self, face):
        """
        Arc beágyazásának kiszámítása
        """
        if face is None:
            return None
            
        face = np.expand_dims(face, axis=0)
        embedding = self.embedding_model.predict(face, verbose=0)
        return embedding[0]
    
    def compute_distance(self, embedding1, embedding2):
        """
        Két beágyazás közötti távolság számítása
        """
        if embedding1 is None or embedding2 is None:
            return float('inf')
            
        return np.linalg.norm(embedding1 - embedding2)

class FaceRecognitionSystem:
    """
    Fő arcfelismerő rendszer osztály
    """
    
    def __init__(self):
        self.data_generator = FaceDataGenerator()
        self.siamese_net = SiameseNetwork()
        self.face_database = {}  # {name: [embedding1, embedding2, ...]}
        self.threshold = 0.7  # Hasonlósági küszöb
        self.recognition_history = []
        
    def enroll_person(self, person_name, num_samples=50):
        """
        Új személy regisztrálása a rendszerbe
        """
        print(f"\n{'='*50}")
        print(f"ÚJ SZEMÉLY REGISZTRÁLÁSA: {person_name}")
        print(f"{'='*50}")
        
        # Arc adatok gyűjtése
        faces = self.data_generator.collect_face_data(person_name, num_samples)
        
        if len(faces) == 0:
            print("Hiba: Nem sikerült arc adatokat gyűjteni!")
            return False
        
        # Beágyazások számítása
        embeddings = []
        for i, face in enumerate(faces):
            embedding = self.siamese_net.compute_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
            print(f"Beágyazás számítása: {i+1}/{len(faces)}")
        
        if len(embeddings) == 0:
            print("Hiba: Nem sikerült beágyazásokat számolni!")
            return False
        
        # Adatbázis frissítése
        self.face_database[person_name] = embeddings
        
        print(f"{person_name} sikeresen regisztrálva!")
        print(f"Beágyazások száma: {len(embeddings)}")
        
        # Adatok mentése
        self.save_database()
        
        return True
    
    def recognize_face(self, face):
        """
        Arc felismerése a regisztrált adatbázis alapján
        """
        if not self.face_database:
            return "ISMERETLEN", 0.0
        
        if face is None:
            return "ISMERETLEN", 0.0
        
        # Beágyazás számítása
        query_embedding = self.siamese_net.compute_embedding(face)
        
        if query_embedding is None:
            return "ISMERETLEN", 0.0
        
        best_match = "ISMERETLEN"
        best_similarity = 0.0
        
        # Hasonlóság számítása minden regisztrált személyre
        for name, embeddings in self.face_database.items():
            for emb in embeddings:
                distance = self.siamese_net.compute_distance(query_embedding, emb)
                similarity = 1.0 / (1.0 + distance)  # Hasonlósági pontszám
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Küszöb alapú döntés
        if best_similarity < self.threshold:
            return "ISMERETLEN", best_similarity
        
        return best_match, best_similarity
    
    def real_time_recognition(self):
        """
        Valós idejű arcfelismerés
        """
        if not self.face_database:
            print("Hiba: Nincsenek regisztrált arcok!")
            return
        
        cap = cv2.VideoCapture(1)  # A te kamerád indexe
        
        if not cap.isOpened():
            print("Hiba: Kamera nem elérhető!")
            return
        
        print(f"\n{'='*50}")
        print("VALÓS IDEJŰ ARCFELISMERÉS INDUL")
        print(f"{'='*50}")
        print("Regisztrált személyek:", list(self.face_database.keys()))
        print("Nyomj 'q'-t a kilépéshez")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            
            # Arc detektálás
            face, bbox = self.data_generator.detect_and_align_face(frame)
            
            if face is not None and bbox is not None:
                x, y, w, h = bbox
                
                # Arc felismerése
                name, confidence = self.recognize_face(face)
                
                # Naplózás
                self.log_recognition(name, confidence)
                
                # Szín beállítása
                color = (0, 255, 0) if name != "ISMERETLEN" else (0, 0, 255)
                
                # Vizualizáció
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display_frame, f'{name} ({confidence:.2f})', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # További információk
                cv2.putText(display_frame, f"Biztonsagi szint: {self.get_security_level(confidence)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Futurisztikus HUD
            self.draw_futuristic_hud(display_frame)
            
            cv2.imshow('Deep Learning Arcfelismero', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_security_level(self, confidence):
        """
        Biztonsági szint meghatározása a megbízhatóság alapján
        """
        if confidence > 0.8:
            return "MAGAS"
        elif confidence > 0.6:
            return "KOZEPES"
        else:
            return "ALACSONY"
    
    def draw_futuristic_hud(self, frame):
        """
        Futurisztikus HUD megjelenítése
        """
        h, w = frame.shape[:2]
        
        # Átlátszó HUD háttér
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Rendszer információk
        cv2.putText(frame, "DEEP LEARNING ARCFELISMERO", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Regisztralt arcok: {len(self.face_database)}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SIAMESE NEURALIS HALO", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Futurisztikus keret
        cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 255, 255), 2)
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (255, 0, 255), 1)
    
    def log_recognition(self, name, confidence):
        """
        Felismerési események naplózása
        """
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'person': name,
            'confidence': float(confidence),
            'security_level': self.get_security_level(confidence)
        }
        self.recognition_history.append(log_entry)
        
        # Csak az első 100 bejegyzést tartjuk meg
        if len(self.recognition_history) > 100:
            self.recognition_history = self.recognition_history[-100:]
        
        # Fájlba írás minden 10. alkalommal
        if len(self.recognition_history) % 10 == 0:
            with open('recognition_log.json', 'w', encoding='utf-8') as f:
                json.dump(self.recognition_history, f, indent=2, ensure_ascii=False)
    
    def save_database(self):
        """
        Arc adatbázis mentése
        """
        # Csak a személyek nevét mentjük, nem a teljes beágyazásokat
        database = {
            'registered_persons': list(self.face_database.keys()),
            'threshold': self.threshold,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(database, f)
        
        print("Adatbázis elmentve!")
    
    def load_database(self):
        """
        Arc adatbázis betöltése
        """
        try:
            with open('face_database.pkl', 'rb') as f:
                database = pickle.load(f)
            
            self.face_database = {name: [] for name in database['registered_persons']}
            self.threshold = database['threshold']
            
            print("Adatbázis betöltve!")
            print(f"Regisztrált személyek: {list(self.face_database.keys())}")
            
        except FileNotFoundError:
            print("Nincs mentett adatbázis! Először regisztrálj személyeket.")
        except Exception as e:
            print(f"Hiba az adatbázis betöltésekor: {e}")
    
    def plot_statistics(self):
        """
        Statisztikák megjelenítése
        """
        if not self.recognition_history:
            print("Nincsenek statisztikai adatok!")
            return
        
        # Hisztogram
        plt.figure(figsize=(10, 6))
        
        names = [log['person'] for log in self.recognition_history]
        unique_names, counts = np.unique(names, return_counts=True)
        
        colors = ['red' if name == 'ISMERETLEN' else 'green' for name in unique_names]
        plt.bar(unique_names, counts, color=colors)
        plt.title('Felismerések megoszlása')
        plt.xticks(rotation=45)
        plt.ylabel('Felismerések száma')
        
        plt.tight_layout()
        plt.savefig('recognition_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Fő program
    """
    print("=== DEEP LEARNING ARCFELISMERŐ RENDSZER ===")
    print("BEADANDÓ PROJEKT - MESTERSEGES INTELLIGENCIA")
    print("SIAMESE NEURÁLIS HÁLÓ ALAPÚ MEGOLDÁS")
    print("=" * 60)
    
    # Rendszer inicializálása
    face_system = FaceRecognitionSystem()
    
    # Siamese háló építése
    print("\nSiamese neurális háló inicializálása...")
    face_system.siamese_net.compile_model()
    
    # Adatbázis betöltése
    face_system.load_database()
    
    while True:
        print("\n=== FŐ MENÜ ===")
        print("1. Új személy regisztrálása")
        print("2. Valós idejű arcfelismerés")
        print("3. Statisztikák megjelenítése")
        print("4. Rendszer információk")
        print("5. Kilépés")
        
        choice = input("Válassz menüpontot (1-5): ")
        
        if choice == '1':
            person_name = input("Add meg a személy nevét: ")
            face_system.enroll_person(person_name, num_samples=50)  # Kevesebb minta a gyorsabb teszteléshez
            
        elif choice == '2':
            face_system.real_time_recognition()
            
        elif choice == '3':
            face_system.plot_statistics()
            
        elif choice == '4':
            print(f"\n=== RENDSZER INFORMÁCIÓK ===")
            print(f"Regisztrált személyek: {list(face_system.face_database.keys())}")
            print(f"Felismerési küszöb: {face_system.threshold}")
            print(f"Naplóbejegyzések: {len(face_system.recognition_history)}")
            print(f"MI algoritmus: Siamese Neurális Háló")
            print(f"Beágyazás dimenzió: 4096")
            print(f"Kamera index: 1")
            
        elif choice == '5':
            print("Kilépés...")
            break
            
        else:
            print("Érvénytelen választás!")

if __name__ == "__main__":
    main()