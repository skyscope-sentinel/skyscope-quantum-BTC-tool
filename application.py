import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import csv
from datetime import datetime

from trading_env import TradingEnv # Your custom trading environment
from rl_agent import QLearningAgent, discretize_state # Your Q-Learning agent

# --- Configuration ---
Q_TABLE_PATH = "q_table.pkl" # Path to the pre-trained Q-table
INITIAL_USDT_BALANCE = Decimal("1000.0") # Starting balance
WINDOW_SIZE = 5 # Must match the agent's training window size
# --- ITERATION POINT: Define how price data is loaded or generated ---

LOG_FILE = "results.csv"

def run(q_table_path, num_episodes=1):
    """
    Runs QLearningAgent.
    Args:
        q_table_path (str): Path to the saved Q-table.
        num_episodes (int): Number of times to run through the price data.
    """
    
    # Initialize Environment with reproducible price data
    # Generate a specific price series for run
    price_generator_env = TradingEnv() # Temporary env just to use its generator
    # For true reproducibility, ensure _generate_dummy_prices can be seeded or load data from a file.
    # np.random.seed(PRICE_DATA_SEED) # Setting seed here affects global numpy state for this call if generator uses np.random directly
    live_prices = price_generator_env._generate_live_prices(
        length=PRICE_DATA_LENGTH, 
        start_price=PRICE_START_PRICE,
        volatility=PRICE_VOLATILITY
    )
    
    env = TradingEnv(
        initial_usdt_balance=INITIAL_USDT_BALANCE,
        price_data=live_prices,
        window_size=WINDOW_SIZE,
        trade_amount_percentage=Decimal("0.1") # Consistent trade size for evaluation
    )
    
    # Initialize Agent and load Q-table
    agent = QLearningAgent(action_space_n=env.action_space_n)
    agent.load_q_table(q_table_path)
    if not agent.q_table: # Check if q_table is empty after loading attempt
        print(f"Q-table at '{q_table_path}' is empty or failed to load. Cannot proceed meaningfully.")
        return

    all_episode_rewards = []
    final_portfolio_values = []
    portfolio_history_all_episodes = []

    print(f"Starting {num_episodes} episode(s)...")

    # Initialize log file
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "step", "timestamp", "action", "price", 
            "usdt_balance", "base_balance", "portfolio_value", "reward",
            "trade_successful"
        ])

    for episode in range(num_episodes):
        observation, info = env.reset() # Seed for env.reset() could be added for more control
        discrete_state = discretize_state(observation, WINDOW_SIZE)
        
        current_episode_reward = Decimal("0.0")
        current_episode_portfolio_history = []
        
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            step_count += 1
            action_idx = agent.choose_action(discrete_state, is_training=False) 
            action_str = ['HOLD', 'BUY', 'SELL'][action_idx]
            
            observation, reward, done, truncated, info = env.step(action_idx)
            next_discrete_state = discretize_state(observation, WINDOW_SIZE)
            
            current_episode_reward += Decimal(str(reward)) # Ensure reward is Decimal
            current_episode_portfolio_history.append(info['portfolio_value'])

            # Log step details
            with open(LIVE_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1, step_count, datetime.now().isoformat(), action_str, 
                    f"{info['current_price']:.2f}", f"{info['usdt_balance']:.2f}", 
                    f"{info['base_balance']:.8f}", f"{info['portfolio_value']:.2f}", f"{reward:.4f}",
                    info.get('trade_successful', False)
                ])
            
            if (step_count % 100 == 0) or (done or truncated): # Log progress periodically
                 print(f"Ep {episode+1}, Step {step_count}: Action: {action_str}, "
                       f"Portfolio: ${info['portfolio_value']:.2f}, Reward: {reward:.4f}")

            discrete_state = next_discrete_state
            if done or truncated: # Check termination conditions
                break
        
        all_episode_rewards.append(current_episode_reward)
        final_portfolio_values.append(info['portfolio_value'])
        portfolio_history_all_episodes.append(current_episode_portfolio_history)
        
        print(f"Episode {episode + 1} finished. Total Reward: {current_episode_reward:.2f}, "
              f"Final Portfolio Value: ${info['portfolio_value']:.2f}")

    print("\n--- Summary ---")
    if not all_episode_rewards:
        print("No episodes were run or completed.")
        return
        
    avg_reward = sum(all_episode_rewards) / len(all_episode_rewards)
    avg_final_portfolio = sum(Decimal(str(p)) for p in final_portfolio_values) / len(final_portfolio_values)
    print(f"Number of Episodes: {num_episodes}")
    print(f"Average Total Reward per Episode: {avg_reward:.2f}")
    print(f"Average Final Portfolio Value: ${avg_final_portfolio:.2f}")
    print(f"Initial Portfolio Value: ${INITIAL_USDT_BALANCE:.2f}")
    
    profit_loss = avg_final_portfolio - INITIAL_USDT_BALANCE
    profit_loss_percent = (profit_loss / INITIAL_USDT_BALANCE) * 100 if INITIAL_USDT_BALANCE > 0 else Decimal("0")
    print(f"Average Profit/Loss: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)")
    print(f"Detailed log saved to: {LOG_FILE}")

    # Plot portfolio value over time for the first episode (if matplotlib available)
    if portfolio_history_all_episodes:
        try:
            plt.figure(figsize=(12,6))
            for i, history in enumerate(portfolio_history_all_episodes):
                if i < 3: # Plot only first few episodes if many
                    plt.plot(history, label=f"Episode {i+1} Portfolio Value")
            
            plt.title("Portfolio Value Over Time (live)")
            plt.xlabel("Step")
            plt.ylabel("Portfolio Value (USDT)")
            plt.legend()
            plt.grid(True)
            plot_filename = "live_portfolio_value.png"
            plt.savefig(plot_filename)
            print(f"Plot of portfolio value saved to {plot_filename}")
            # plt.show() # Uncomment to display plot if running in suitable environment
        except ImportError:
            print("Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'.")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == '__main__':
    # Ensure a Q-table exists from training (rl_agent.py) before running live.
    # This basic check helps the user.
    if not os.path.exists(Q_TABLE_PATH):
        print(f"Error: Q-table not found at '{Q_TABLE_PATH}'.")
        print("Please run rl_agent.py to train and save a Q-table first.")
    else:
        # Set a global seed for numpy before generating price data for this run,
        # to make the _generate_dummy_prices more reproducible if it uses np.random directly
        # without its own internal seeding mechanism.
        np.random.seed(PRICE_DATA_SEED) 
        run_live(q_table_path=Q_TABLE_PATH, num_episodes=1) # Run for 1 episode for a quick test
        # For multiple independent live runs with different data, you might vary PRICE_DATA_SEED
        # or preferably load different pre-generated datasets.
        # np.random.seed(PRICE_DATA_SEED + 1) # Example for a second, different run
        # run_live(q_table_path=Q_TABLE_PATH, num_episodes=5)
