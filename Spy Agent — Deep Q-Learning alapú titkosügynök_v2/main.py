import argparse
from train import train_agent
from test_agent import test_agent

def main():
    parser = argparse.ArgumentParser(description='DQN Spy Agent')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'both'],
                       help='Mód: train (tanítás), test (teszt), both (mindkettő)')
    parser.add_argument('--episodes', type=int, default=1500,
                       help='Tanítási epizódok száma')
    parser.add_argument('--test_episodes', type=int, default=5,
                       help='Teszt epizódok száma')
    parser.add_argument('--delay', type=float, default=0.2,
                       help='Késleltetés tesztelésnél (másodperc)')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'both':
        print("╔══════════════════════════════════════╗")
        print("║           TANÍTÁS MÓD               ║")
        print("╚══════════════════════════════════════╝")
        train_agent()
    
    if args.mode == 'test' or args.mode == 'both':
        print("\n╔══════════════════════════════════════╗")
        print("║            TESZT MÓD                ║")
        print("╚══════════════════════════════════════╝")
        success_rate, avg_reward = test_agent(
            episodes=args.test_episodes, 
            render=True, 
            delay=args.delay
        )

if __name__ == "__main__":
    main()