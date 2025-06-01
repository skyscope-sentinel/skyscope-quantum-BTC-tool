import numpy as np
import pickle # For saving/loading Q-table
import os
from collections import defaultdict
from trading_env import TradingEnv # Import the custom environment
from decimal import Decimal

# Helper function to discretize continuous state from TradingEnv
# This is a crucial and often tricky part for Q-learning with continuous states.
# The number of bins per feature will determine the size of the Q-table.
# Example: obs = [p1, p2, p3, p4, p5, usdt_bal_norm, base_bal_norm]
# Price features are normalized around 0 (e.g. -0.1 to 0.1 perhaps)
# Balance features are normalized 0 to 1.

# Define discretization bins more carefully
# Prices are (price/mean_price_in_window) - 1. So range could be e.g. -0.2 to 0.2
PRICE_BINS = np.linspace(-0.3, 0.3, num=5) # 5 bins for each price point in window (e.g. very low, low, mid, high, very high relative to mean)
USDT_BALANCE_BINS = np.linspace(0, 1, num=5) # 5 bins for USDT balance %
BASE_BALANCE_BINS = np.linspace(0, 1, num=5) # 5 bins for Base currency balance % (relative to max possible)

def discretize_state(observation, window_size):
    """
    Converts a continuous observation array from TradingEnv into a discrete state tuple.
    Args:
        observation (np.array): The continuous observation.
                                Expected shape: (window_size + 2,)
        window_size (int): The number of price points in the window.
    Returns:
        tuple: A tuple of integers representing the discrete state.
    """
    discrete_obs = []
    # Discretize price window components
    for i in range(window_size):
        discrete_obs.append(int(np.digitize(observation[i], PRICE_BINS)))
    
    # Discretize normalized USDT balance
    discrete_obs.append(int(np.digitize(observation[window_size], USDT_BALANCE_BINS)))
    # Discretize normalized Base currency balance
    discrete_obs.append(int(np.digitize(observation[window_size + 1], BASE_BALANCE_BINS)))
    
    return tuple(discrete_obs)


class QLearningAgent:
    def __init__(self, action_space_n, state_space_dims_bins=None, learning_rate=0.1, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000):
        """
        Simple Q-Learning Agent.
        Args:
            action_space_n (int): Number of possible actions.
            state_space_dims_bins (list of int): Not directly used if Q-table is defaultdict.
                                                 Useful for knowing Q-table size if pre-allocated.
            learning_rate (float): Alpha, the learning rate.
            discount_factor (float): Gamma, discount factor for future rewards.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay_steps (int): Number of steps over which epsilon decays.
        """
        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        # Calculate decay rate such that epsilon reaches epsilon_end after epsilon_decay_steps
        # One common way: epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-decay_rate * step)
        # Here, simple linear decay or step-based decay.
        # For simplicity, let's use a multiplicative decay per step, or rather, per call to learn() or choose_action()
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.current_decay_step = 0

        # Q-table: using defaultdict for convenience, so states are added as encountered.
        # Keys are discrete state tuples, values are numpy arrays of Q-values for each action.
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))

    def choose_action(self, discrete_state, is_training=True):
        """
        Chooses an action using epsilon-greedy strategy.
        """
        if is_training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_n)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value for this state
            # If all Q-values are zero for this state (new state), pick randomly to explore.
            if np.sum(self.q_table[discrete_state]) == 0:
                 return np.random.randint(self.action_space_n)
            return np.argmax(self.q_table[discrete_state])

    def learn(self, discrete_state, action, reward, next_discrete_state, done):
        """
        Updates the Q-table using the Bellman equation.
        Q(s, a) = Q(s, a) + lr * (reward + gamma * max_Q(s', a') - Q(s, a))
        """
        current_q = self.q_table[discrete_state][action]
        
        # If done, the future reward (max_Q(s',a')) is 0 because there's no next state.
        max_future_q = np.max(self.q_table[next_discrete_state]) if not done else 0.0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        
        self.q_table[discrete_state][action] = new_q

        # Decay epsilon (simple linear decay per learning step)
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_rate
            # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate) # Multiplicative decay
        self.current_decay_step +=1


    def save_q_table(self, filepath="q_table.pkl"):
        # defaultdict cannot be pickled directly if it contains lambdas that are not top-level.
        # Convert to regular dict before saving.
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"Q-table saved to {filepath}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filepath="q_table.pkl"):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
                    self.q_table.update(loaded_dict)
                print(f"Q-table loaded from {filepath}")
            else:
                print(f"No Q-table found at {filepath}. Starting with a new one.")
        except Exception as e:
            print(f"Error loading Q-table: {e}. Starting with a new one.")


# Example Training Loop
if __name__ == '__main__':
    WINDOW_SIZE = 5 # Must match env's window_size for discretize_state
    
    # Initialize environment
    # Using a smaller dataset for faster example training
    dummy_prices = TradingEnv()._generate_dummy_prices(length=500, start_price=Decimal("200")) 
    env = TradingEnv(initial_usdt_balance=Decimal("100.0"), 
                     price_data=dummy_prices, 
                     window_size=WINDOW_SIZE,
                     trade_amount_percentage=Decimal("0.5")) # Trade 50% for more impact in training

    # Initialize Q-Learning Agent
    # State space size definition is tricky with dynamic defaultdict Q-table.
    # For info: num_price_bins^window_size * num_usdt_bins * num_base_bins
    agent = QLearningAgent(action_space_n=env.action_space_n,
                           epsilon_decay_steps=2000) # Decay epsilon over more steps

    agent.load_q_table() # Try to load existing Q-table

    num_episodes = 1000 # Number of training episodes
    max_steps_per_episode = env.max_steps 
    
    episode_rewards = []

    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        observation, info = env.reset()
        discrete_state = discretize_state(observation, WINDOW_SIZE)
        
        current_episode_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.choose_action(discrete_state)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_discrete_state = discretize_state(next_observation, WINDOW_SIZE)
            
            agent.learn(discrete_state, action, reward, next_discrete_state, terminated or truncated)
            
            current_episode_reward += reward
            discrete_state = next_discrete_state
            
            if terminated or truncated:
                break
        
        episode_rewards.append(current_episode_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} finished. Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            # Save Q-table periodically
            agent.save_q_table()

    print("Training finished.")
    agent.save_q_table() # Save final Q-table

    # --- Example of running the trained agent (exploitation mode) ---
    print("\n--- Running trained agent (exploitation mode) ---")
    observation, info = env.reset()
    discrete_state = discretize_state(observation, WINDOW_SIZE)
    env.render()
    total_reward_test = 0
    for _ in range(max_steps_per_episode):
        action = agent.choose_action(discrete_state, is_training=False) # is_training=False for exploitation
        print(f"State: {discrete_state}, Action: {['HOLD', 'BUY', 'SELL'][action]}")
        
        observation, reward, terminated, truncated, info = env.step(action)
        discrete_state = discretize_state(observation, WINDOW_SIZE)
        env.render()
        total_reward_test += reward
        if terminated or truncated:
            break
    print(f"Total reward in test run: {total_reward_test:.4f}")
    print("Final portfolio value:", info.get('portfolio_value'))
