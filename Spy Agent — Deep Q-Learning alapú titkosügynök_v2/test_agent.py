import numpy as np
from env import SpyEnv
from dqn import DQNAgent
import time

def test_agent(episodes=10, render=True, delay=0.3):
    env = SpyEnv(size=8)
    state_size = len(env._get_state())
    action_size = env.action_space
    
    agent = DQNAgent(state_size, action_size)
    agent.load("spy_agent_best")  # legjobb modell betöltése
    
    # Epsilon 0 teszteléshez
    agent.epsilon = 0.0
    
    total_rewards = []
    success_count = 0
    
    print("╔══════════════════════════════════════╗")
    print("║          TESZTELÉS INDUL            ║")
    print("╚══════════════════════════════════════╝")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        if render:
            print(f"\n╔══════════════════════════════════════╗")
            print(f"║         TESZT EPIZÓD {episode + 1:2d}            ║")
            print(f"╚══════════════════════════════════════╝")
            env.render()
            time.sleep(delay)
        
        for step in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            if done:
                if reward > 1.0:  # Sikeres epizód
                    success_count += 1
                    print(f"╔══════════════════════════════════════╗")
                    print(f"║    ✓ CÉL ELÉRVE!                     ║")
                    print(f"║    Lépések: {steps:3d}                     ║")
                    print(f"║    Jutalom: {total_reward:6.2f}               ║")
                    print(f"╚══════════════════════════════════════╝")
                else:
                    print(f"╔══════════════════════════════════════╗")
                    print(f"║    ✗ SIKERTELEN                      ║")
                    print(f"║    Lépések: {steps:3d}                     ║")
                    print(f"║    Jutalom: {total_reward:6.2f}               ║")
                    print(f"╚══════════════════════════════════════╝")
                break
        
        total_rewards.append(total_reward)
    
    # Statisztikák
    success_rate = success_count / episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║         TESZT EREDMÉNYEK            ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║ Epizódok: {episodes:3d}                      ║")
    print(f"║ Sikerráta: {success_rate:6.1%}                 ║")
    print(f"║ Átlagos jutalom: {avg_reward:6.2f} ± {std_reward:4.2f}   ║")
    print(f"║ Sikeres epizódok: {success_count:2d}/{episodes:2d}               ║")
    print(f"╚══════════════════════════════════════╝")
    
    return success_rate, avg_reward

if __name__ == "__main__":
    test_agent(episodes=5, render=True, delay=0.2)