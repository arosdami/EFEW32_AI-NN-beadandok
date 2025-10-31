import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        
        # Hiperparaméterek
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Modell építése
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.learning_steps = 0
        
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.array([state])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Jelenlegi Q-értékek
        current_qs = self.model.predict(states, verbose=0)
        
        # Jövőbeli Q-értékek
        future_qs = self.target_model.predict(next_states, verbose=0)
        
        # Cél Q-értékek
        target_qs = current_qs.copy()
        
        for i in range(len(minibatch)):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                target_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(future_qs[i])
        
        # Modell tanítása
        self.model.fit(states, target_qs, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # Epsilon csökkentése
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.learning_steps += 1
        
        # Célháló frissítése
        if self.learning_steps % 100 == 0:
            self.update_target_model()
    
    def save(self, name):
        self.model.save(f"{name}.keras")
        self.target_model.save(f"{name}_target.keras")
    
    def load(self, name):
        try:
            self.model = tf.keras.models.load_model(f"{name}.keras")
            self.target_model = tf.keras.models.load_model(f"{name}_target.keras")
            print(f"Model loaded successfully: {name}")
        except:
            print(f"Could not load model: {name}, starting from scratch")