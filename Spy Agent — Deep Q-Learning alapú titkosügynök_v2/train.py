import numpy as np
import matplotlib.pyplot as plt
from env import SpyEnv
from dqn import DQNAgent
import os

def train_agent():
    # Környezet és agent inicializálása
    env = SpyEnv(size=8)
    state_size = len(env._get_state())
    action_size = env.action_space
    
    print(f"Állapot méret: {state_size}, Akciók száma: {action_size}")
    
    agent = DQNAgent(state_size, action_size)
    
    # Tanítási paraméterek
    episodes = 1500
    replay_interval = 4
    
    # Metrikák nyomon követése
    rewards_history = []
    steps_history = []
    epsilon_history = []
    success_rate = []
    
    # Checkpoint mappa
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    best_avg_reward = -float('inf')
    
    print("Tanítás megkezdése...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            # Akció választása és végrehajtása
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # Emlékezet frissítése
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Periodikus tanulás
            if step % replay_interval == 0:
                agent.replay()
            
            if done:
                break
        
        # Utolsó replay az epizód végén
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Metrikák frissítése
        rewards_history.append(total_reward)
        steps_history.append(steps)
        epsilon_history.append(agent.epsilon)
        
        # Sikerráta számítása (pozitív jutalom = siker)
        recent_rewards = rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history
        success_count = sum(1 for r in recent_rewards if r > 1.0)
        current_success_rate = success_count / len(recent_rewards)
        success_rate.append(current_success_rate)
        
        # Progress jelentés
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_steps = np.mean(steps_history[-50:])
            print(f"Epizód {episode+1}/{episodes}, "
                  f"Átlag jutalom: {avg_reward:.2f}, "
                  f"Átlag lépések: {avg_steps:.1f}, "
                  f"ε: {agent.epsilon:.3f}, "
                  f"Sikerráta: {current_success_rate:.2f}")
        
        # Legjobb modell mentése
        if len(rewards_history) >= 100:
            recent_avg = np.mean(rewards_history[-100:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                agent.save("spy_agent_best")
                print(f"Új legjobb modell! Átlag jutalom: {best_avg_reward:.2f}")
        
        # Checkpoint mentése
        if (episode + 1) % 200 == 0:
            checkpoint_path = f"checkpoints/spy_agent_episode_{episode+1}"
            agent.save(checkpoint_path)
            print(f"Checkpoint mentve: {checkpoint_path}")
    
    # Végső modell mentése
    agent.save("spy_agent_final")
    
    # Metrikák ábrázolása
    plot_training_metrics(rewards_history, steps_history, epsilon_history, success_rate)
    
    return agent, rewards_history

def plot_training_metrics(rewards, steps, epsilon, success_rate):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Jutalom
    ax1.plot(rewards, alpha=0.6)
    ax1.set_title('Jutalom epizódonként')
    ax1.set_xlabel('Epizód')
    ax1.set_ylabel('Jutalom')
    ax1.grid(True)
    
    # Mozgóátlag jutalom
    window = 50
    if len(rewards) > window:
        moving_avg = [np.mean(rewards[i:i+window]) for i in range(len(rewards)-window)]
        ax1.plot(range(window, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Mozgóátlag ({window} ep.)')
        ax1.legend()
    
    # Lépések
    ax2.plot(steps, alpha=0.6, color='green')
    ax2.set_title('Lépések epizódonként')
    ax2.set_xlabel('Epizód')
    ax2.set_ylabel('Lépések')
    ax2.grid(True)
    
    # Epsilon
    ax3.plot(epsilon, color='orange')
    ax3.set_title('Epsilon értéke')
    ax3.set_xlabel('Epizód')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True)
    
    # Sikerráta
    ax4.plot(success_rate, color='purple')
    ax4.set_title('Sikerráta (100 ep. mozgóátlag)')
    ax4.set_xlabel('Epizód')
    ax4.set_ylabel('Sikerráta')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    agent, history = train_agent()
    print("Tanítás befejezve!")