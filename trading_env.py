import numpy as np
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext

# Set precision for Decimal calculations
getcontext().prec = 28 # Ensure high precision for financial calcs

# Assuming these constants might be shared or configured elsewhere eventually
# For now, define them here for simplicity of the environment
TRADING_SYMBOL = "BTCUSDT"
BASE_CURRENCY = TRADING_SYMBOL.replace("USDT", "")
QUOTE_CURRENCY = "USDT"

MIN_TRADE_QTY_BASE = Decimal("0.00001") # Min BTC to trade
MIN_TRADE_VALUE_QUOTE = Decimal("1.0") # Min USDT value of a trade
# --- ITERATION POINT: Transaction fees impact profitability significantly ---
TRANSACTION_FEE_PERCENT = Decimal("0.001") # 0.1% transaction fee (example for Bybit taker fee)

class TradingEnv:
    def __init__(self, initial_usdt_balance=Decimal("100.0"), price_data=None, window_size=5, trade_amount_percentage=Decimal("0.1")):
        """
        Simplified Trading Environment for a Reinforcement Learning Agent.

        Args:
            initial_usdt_balance (Decimal): Starting balance in USDT.
            price_data (list or np.array): A series of prices for the asset. 
                                           If None, a simple random walk is generated.
            window_size (int): Number of past price points to include in the observation.
            trade_amount_percentage (Decimal): Percentage of available balance/holdings to trade.
        """
        self.initial_usdt_balance = initial_usdt_balance.quantize(Decimal('0.01'))
        self.window_size = window_size 
        self.trade_amount_percentage = trade_amount_percentage

        if price_data is not None and len(price_data) > self.window_size :
            self.price_data = [Decimal(p) for p in price_data]
        else:
            self.price_data = self._generate_dummy_prices(length=200, start_price=Decimal("50000"))
        
        # Ensure max_steps is not negative if price_data is too short
        self.max_steps = max(0, len(self.price_data) - 1 - self.window_size)

        # Portfolio state
        self.current_usdt_balance = self.initial_usdt_balance
        self.current_base_balance = Decimal("0.0")
        
        self.current_step = 0
        # Start at the first point where a full window is available
        self.current_price_index = self.window_size 

        # Action space: 0: HOLD, 1: BUY, 2: SELL
        self.action_space_n = 3 
        # Observation space shape (normalized windowed prices + normalized balances)
        self.observation_space_shape = (self.window_size + 2,) 

        self.log = [] 

    def _generate_dummy_prices(self, length=200, start_price=Decimal("50000"), volatility=Decimal("0.005")):
        prices = [start_price.quantize(Decimal('0.01'))]
        for _ in range(1, length):
            # Simulate a somewhat realistic price movement (mean-reverting tendency + random noise)
            drift = Decimal(np.random.normal(0, 0.0001)) # Small random drift
            reversion = (start_price - prices[-1]) * Decimal('0.001') # Tendency to revert to start_price
            noise = Decimal(np.random.normal(0, float(volatility)))
            change_percent = drift + reversion + noise
            new_price = prices[-1] * (1 + change_percent)
            prices.append(new_price.quantize(Decimal("0.01")))
        return prices

    def _get_observation(self):
        if self.current_price_index < self.window_size:
            # This case should ideally be handled by ensuring price_data is long enough
            # or by padding at the beginning of self.price_data during init.
            # For simplicity here, if it occurs, pad with the earliest available price.
            padding_needed = self.window_size - self.current_price_index
            price_window_segment = self.price_data[0:self.current_price_index]
            price_window = ([self.price_data[0]] * padding_needed) + price_window_segment
        else:
            price_window = self.price_data[self.current_price_index - self.window_size : self.current_price_index]
        
        # Normalize price window by its own mean to capture relative changes
        mean_price_in_window = sum(price_window) / len(price_window) if len(price_window) > 0 else Decimal('1')
        norm_price_window = [(p / mean_price_in_window) - Decimal('1') for p in price_window] # Normalize around 0
        
        # Normalize balances: current_usdt / initial_usdt
        norm_usdt_balance = self.current_usdt_balance / self.initial_usdt_balance
        # norm_base_balance: current_base / (initial_usdt / typical_price)
        # Use a typical price like the first price for scaling base balance normalization
        typical_price = self.price_data[0] if len(self.price_data) > 0 else Decimal('50000')
        max_possible_base = self.initial_usdt_balance / typical_price if typical_price > 0 else Decimal('1')
        norm_base_balance = self.current_base_balance / max_possible_base if max_possible_base > 0 else Decimal('0')

        obs_list = [float(p) for p in norm_price_window] + [float(norm_usdt_balance), float(norm_base_balance)]
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Potentially re-generate prices if they are meant to be stochastic per episode
        # For now, assume price_data is fixed per TradingEnv instance after init.
        # self.price_data = self._generate_dummy_prices() # Uncomment for new prices each reset

        self.current_usdt_balance = self.initial_usdt_balance
        self.current_base_balance = Decimal("0.0")
        self.current_step = 0
        self.current_price_index = self.window_size
        self.log = []
        
        return self._get_observation(), {}

    def _calculate_reward(self, previous_portfolio_value, current_portfolio_value, action_taken, trade_successful):
        # --- ITERATION POINT: Reward function design is critical for RL agent behavior ---
        reward = current_portfolio_value - previous_portfolio_value # Profit or loss
        
        if not trade_successful and action_taken != 0: # Penalize failed trade attempts
            reward -= Decimal("0.1") * self.initial_usdt_balance * Decimal("0.001") # Small penalty relative to initial capital

        # Optional: Penalty for holding if market is volatile (encourages participation)
        # Or, reward for holding in very risky situations.
        # if action_taken == 0:
        #     reward -= Decimal("0.0001") # Small cost of doing nothing

        return float(reward)


    def step(self, action): # action: 0=HOLD, 1=BUY, 2=SELL
        self.current_step += 1
        # current_price_index is the index of the price that is current *after* this step.
        # The action is based on the price *before* this step.
        price_at_decision = self.price_data[self.current_price_index -1] 
        
        previous_portfolio_value = (self.current_usdt_balance + self.current_base_balance * price_at_decision).quantize(Decimal("0.01"))
        
        trade_successful_this_step = False

        if action == 1: # BUY BASE CURRENCY
            usdt_to_spend = (self.current_usdt_balance * self.trade_amount_percentage).quantize(Decimal("0.01"), ROUND_DOWN)
            if self.current_usdt_balance >= MIN_TRADE_VALUE_QUOTE and usdt_to_spend >= MIN_TRADE_VALUE_QUOTE :
                quantity_to_buy_before_fees = (usdt_to_spend / price_at_decision).quantize(Decimal("0.00000001"), ROUND_DOWN)
                fee = (quantity_to_buy_before_fees * TRANSACTION_FEE_PERCENT).quantize(Decimal("0.00000001"), ROUND_UP)
                quantity_to_buy_after_fees = quantity_to_buy_before_fees - fee

                if quantity_to_buy_after_fees >= MIN_TRADE_QTY_BASE:
                    self.current_usdt_balance -= usdt_to_spend 
                    self.current_base_balance += quantity_to_buy_after_fees
                    trade_successful_this_step = True
                    self.log.append(f"Step {self.current_step}: BUY {quantity_to_buy_after_fees:.8f} {BASE_CURRENCY} at {price_at_decision:.2f}")
                else: self.log.append(f"Step {self.current_step}: BUY fail (qty {quantity_to_buy_after_fees:.8f} < min or fee too high)")
            else: self.log.append(f"Step {self.current_step}: BUY fail (USDT avail {self.current_usdt_balance:.2f} or to spend {usdt_to_spend:.2f} < min value {MIN_TRADE_VALUE_QUOTE})")
        
        elif action == 2: # SELL BASE CURRENCY
            base_to_sell = (self.current_base_balance * self.trade_amount_percentage).quantize(Decimal("0.00000001"), ROUND_DOWN)
            if base_to_sell >= MIN_TRADE_QTY_BASE:
                value_in_usdt_before_fees = (base_to_sell * price_at_decision).quantize(Decimal("0.01"), ROUND_DOWN)
                
                if value_in_usdt_before_fees >= MIN_TRADE_VALUE_QUOTE:
                    fee = (value_in_usdt_before_fees * TRANSACTION_FEE_PERCENT).quantize(Decimal("0.01"), ROUND_UP)
                    value_in_usdt_after_fees = value_in_usdt_before_fees - fee
                    
                    self.current_base_balance -= base_to_sell
                    self.current_usdt_balance += value_in_usdt_after_fees
                    trade_successful_this_step = True
                    self.log.append(f"Step {self.current_step}: SELL {base_to_sell:.8f} {BASE_CURRENCY} at {price_at_decision:.2f}, net USDT {value_in_usdt_after_fees:.2f}")
                else: self.log.append(f"Step {self.current_step}: SELL fail (value {value_in_usdt_before_fees:.2f} < min value {MIN_TRADE_VALUE_QUOTE})")
            else: self.log.append(f"Step {self.current_step}: SELL fail (qty {base_to_sell:.8f} < min qty {MIN_TRADE_QTY_BASE})")
        else: # HOLD
            self.log.append(f"Step {self.current_step}: HOLD at {price_at_decision:.2f}")

        # Update price index for the next state
        self.current_price_index +=1
        price_after_action = self.price_data[self.current_price_index -1] 

        current_portfolio_value = (self.current_usdt_balance + self.current_base_balance * price_after_action).quantize(Decimal("0.01"))
        reward = self._calculate_reward(previous_portfolio_value, current_portfolio_value, action, trade_successful_this_step)
        
        terminated = False 
        if self.current_usdt_balance < Decimal("0.01") and (self.current_base_balance * price_after_action) < MIN_TRADE_VALUE_QUOTE :
            terminated = True
            reward -= 1000 # Heavy penalty for going broke

        truncated = False
        if self.current_step >= self.max_steps or self.current_price_index >= len(self.price_data):
            truncated = True
            
        observation = self._get_observation()
        info = {
            'current_price': float(price_after_action),
            'usdt_balance': float(self.current_usdt_balance),
            'base_balance': float(self.current_base_balance),
            'portfolio_value': float(current_portfolio_value),
            'trade_successful': trade_successful_this_step
        }
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        price_to_render = self.price_data[self.current_price_index -1] # Current price for this state
        portfolio_val = (self.current_usdt_balance + self.current_base_balance * price_to_render).quantize(Decimal("0.01"))
        print(f"Step: {self.current_step}, Price: {price_to_render:.2f}, "
              f"USDT: {self.current_usdt_balance:.2f}, {BASE_CURRENCY}: {self.current_base_balance:.8f}, "
              f"Portfolio Value: {portfolio_val:.2f}")

# Example usage:
if __name__ == '__main__':
    env = TradingEnv(initial_usdt_balance=Decimal("100.0"), trade_amount_percentage=Decimal("1.0")) # Trade 100% for testing
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space_shape}")
    print(f"Action space size: {env.action_space_n}")
    print("Initial Observation:", obs)
    env.render()

    total_reward = 0
    for i in range(env.max_steps + 5): # Try to run a bit beyond max_steps to test truncation
        action = np.random.randint(0, env.action_space_n) 
        # action = 1 # Force BUY to test logic
        print(f"--- Taking action: {['HOLD', 'BUY', 'SELL'][action]} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Reward: {reward:.4f}, Info: {info}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps.")
            if terminated: print("Reason: Agent went broke or similar terminal condition.")
            if truncated: print("Reason: Max steps reached or end of data.")
            break
    print(f"Total reward over episode: {total_reward:.4f}")
    # print("\nTrade Log:")
    # for entry in env.log:
    #    print(entry)
