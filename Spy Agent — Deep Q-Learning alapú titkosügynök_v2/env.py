import numpy as np
import random
import time
import os

class SpyEnv:
    def __init__(self, size=8):
        self.size = size
        self.action_space = 4  # 0=fel, 1=le, 2=balra, 3=jobbra
        self.reset()
        
    def reset(self):
        # Agent kezdőpozíció
        self.agent_pos = [0, 0]
        
        # Cél pozíció - mindig ellentétes sarok
        self.goal_pos = [self.size-1, self.size-1]
        
        # Akadályok generálása
        self.obstacles = self._generate_obstacles(num_obstacles=self.size)
        
        self.steps = 0
        self.max_steps = 100
        
        return self._get_state()
    
    def _generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                # Ne legyen az agent vagy cél pozícióban, és legyen elég távol a kezdőponttól
                if (pos != self.agent_pos and pos != self.goal_pos and 
                    pos not in obstacles and
                    abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1]) > 2):
                    obstacles.append(pos)
                    break
        return obstacles
    
    def _get_state(self):
        # Állapot: agent pozíció, cél pozíció, legközelebbi akadály távolsága
        state = [
            self.agent_pos[0] / self.size,  # x normalizálva
            self.agent_pos[1] / self.size,  # y normalizálva
            (self.goal_pos[0] - self.agent_pos[0]) / self.size,  # cél relatív x
            (self.goal_pos[1] - self.agent_pos[1]) / self.size,  # cél relatív y
        ]
        
        # Legközelebbi akadály távolsága mind a 4 irányban
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            dist = self._get_obstacle_distance(dx, dy)
            state.append(dist / self.size)
        
        return np.array(state, dtype=np.float32)
    
    def _get_obstacle_distance(self, dx, dy):
        """Visszaadja a legközelebbi akadály távolságát egy adott irányban"""
        x, y = self.agent_pos
        distance = 0
        
        while 0 <= x + dx < self.size and 0 <= y + dy < self.size:
            x += dx
            y += dy
            distance += 1
            if [x, y] in self.obstacles:
                return distance
            if [x, y] == self.goal_pos:
                return distance  # Cél is akadálynak számít a távolság számításnál
        
        return distance
    
    def step(self, action):
        self.steps += 1
        old_pos = self.agent_pos.copy()
        
        # Akció végrehajtása
        if action == 0 and self.agent_pos[1] > 0:  # fel
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:  # le
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # balra
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size - 1:  # jobbra
            self.agent_pos[0] += 1
        
        # Jutalom számítás
        reward = -0.01  # alap költség
        
        # Cél elérése
        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10.0  # Nagyobb jutalom a célért
        # Akadályba ütközés
        elif self.agent_pos in self.obstacles:
            reward = -5.0
            done = True
        # Túl sok lépés
        elif self.steps >= self.max_steps:
            reward = -2.0
            done = True
        # Közeledés a célhoz
        else:
            old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
            new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            if new_dist < old_dist:
                reward += 0.1  # nagyobb jutalom a cél felé haladásért
            elif new_dist > old_dist:
                reward -= 0.05  # büntetés a céltól távolodásért
        
        return self._get_state(), reward, done, {}
    
    def render(self, clear_screen=True):
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
            
        grid = np.full((self.size, self.size), '·')  # Üres mezők
        
        # Akadályok
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = '█'
        
        # Cél
        grid[self.goal_pos[1]][self.goal_pos[0]] = 'G'
        
        # Agent
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
        
        print("╔" + "═" * (self.size * 2 + 1) + "╗")
        print(f"║ Lépés: {self.steps:2d}/{self.max_steps} {' ' * (self.size * 2 - 15)}║")
        print("╠" + "═" * (self.size * 2 + 1) + "╣")
        
        for i, row in enumerate(grid):
            print("║ " + ' '.join(row) + " ║")
        
        print("╚" + "═" * (self.size * 2 + 1) + "╝")
        print("Jelmagyarázat: A = Ügynök, G = Cél, █ = Akadály, · = Út")
        print()