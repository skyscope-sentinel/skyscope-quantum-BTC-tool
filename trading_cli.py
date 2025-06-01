import argparse
import json
import os
import logging
import requests # For Ollama API
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
import csv 
from datetime import datetime
import numpy as np # For RL agent state and actions

# Custom module imports
from pybit.unified_trading import HTTP
# Qiskit (placeholders, assuming QuantumInspiredOptimizer is a separate class now if used)
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
# from qiskit_aer import AerSimulator 
# from qiskit.exceptions import QiskitError 

# RL Agent imports
from trading_env import TradingEnv # Used for constants like RL_ENV_WINDOW_SIZE if defined there, and for understanding structure
from rl_agent import QLearningAgent, discretize_state # Import the agent and discretization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
CONFIG_FILE = "config.json"
TRADE_LOG_FILE = "trade_log.csv"
PORTFOLIO_STATE_FILE = "portfolio_state.json"
DEFAULT_RL_Q_TABLE_PATH = "q_table.pkl" 

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"

MIN_BTC_ORDER_QTY = Decimal("0.00001")
MIN_USDT_ORDER_VALUE = Decimal("1.0")
DEFAULT_USDT_TRADE_PERCENTAGE = Decimal("0.03")
TRADING_SYMBOL = "BTCUSDT"
BASE_CURRENCY = TRADING_SYMBOL.replace("USDT", "")
QUOTE_CURRENCY = "USDT"
# --- ITERATION POINT: This window size must match the one used for training the RL agent ---
RL_ENV_WINDOW_SIZE = 5 

getcontext().prec = 28

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {CONFIG_FILE}")

def init_trade_log():
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "action", "price", "quantity_base", "value_quote",
                "agent_type", "agent_raw_advice", "agent_action", "agent_qty_suggestion",
                "quantum_summary", "order_id", "status",
                "portfolio_usdt_before", "portfolio_base_before",
                "portfolio_usdt_after", "portfolio_base_after", "cycle_id"
            ])

def log_trade_action(data_dict):
    headers = [
        "timestamp", "symbol", "action", "price", "quantity_base", "value_quote",
        "agent_type", "agent_raw_advice", "agent_action", "agent_qty_suggestion",
        "quantum_summary", "order_id", "status",
        "portfolio_usdt_before", "portfolio_base_before",
        "portfolio_usdt_after", "portfolio_base_after", "cycle_id"
    ]
    for header in headers: data_dict.setdefault(header, "N/A")
    
    is_empty = not os.path.exists(TRADE_LOG_FILE) or os.stat(TRADE_LOG_FILE).st_size == 0
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if is_empty: writer.writeheader()
        writer.writerow(data_dict)

def load_portfolio_state():
    if os.path.exists(PORTFOLIO_STATE_FILE):
        with open(PORTFOLIO_STATE_FILE, 'r') as f:
            state = json.load(f)
            state[QUOTE_CURRENCY] = Decimal(state.get(QUOTE_CURRENCY, "3.0"))
            state[BASE_CURRENCY] = Decimal(state.get(BASE_CURRENCY, "0.0"))
            return state
    initial_usdt = Decimal(load_config().get("initial_usdt_balance", "3.0"))
    return {QUOTE_CURRENCY: initial_usdt, BASE_CURRENCY: Decimal("0.0")}

def save_portfolio_state(state):
    serializable_state = {k: str(v.quantize(Decimal('0.0000000001')) if isinstance(v, Decimal) else str(v)) for k, v in state.items()}
    with open(PORTFOLIO_STATE_FILE, 'w') as f:
        json.dump(serializable_state, f, indent=4)

# --- Dummy QuantumInspiredOptimizer for now if not properly defined elsewhere ---
# This should ideally be imported if it's in a separate file from the original plan
class QuantumInspiredOptimizer:
    def __init__(self, n_qubits=5, shots=100):
        self.n_qubits = n_qubits
        self.shots = shots
        logger.info("QuantumInspiredOptimizer initialized (EXPERIMENTAL/PLACEHOLDER: Not used for actual trading decisions in this version).")

    def generate_parameters(self):
        logger.info("QuantumInspiredOptimizer.generate_parameters called (EXPERIMENTAL/PLACEHOLDER: Returning None, no parameters generated).")
        return None


class ByBitTrader:
    def __init__(self, api_key, api_secret, testnet=True):
        self.is_testnet = testnet
        try:
            self.session = HTTP(testnet=self.is_testnet, api_key=api_key, api_secret=api_secret)
            logger.info(f"ByBit session initialized. Testnet: {self.is_testnet}")
        except Exception as e: logger.error(f"ByBit init error: {e}"); self.session = None # Ensure session is None on failure

    def get_wallet_balance(self, account_type="UNIFIED", coin="USDT"):
        if not self.session: logger.error("Bybit session not available for get_wallet_balance."); return None
        try:
            r = self.session.get_wallet_balance(accountType=account_type, coin=coin)
            if r and r.get('retCode') == 0:
                for b_info in r.get('result', {}).get('list', []):
                    if b_info.get('accountType') == account_type or account_type == "ANY":
                        for c_bal in b_info.get('coin', []):
                            if c_bal.get('coin') == coin: return Decimal(c_bal.get('walletBalance', "0"))
                logger.warning(f"Could not find {coin} in {account_type} from ByBit API response.")
                return Decimal("0") # Return 0 if specific coin not found but call was successful
            logger.error(f"ByBit API Error get_wallet_balance: {r.get('retMsg') if r else 'No response'}")
            return None
        except Exception as e: logger.error(f"Exception in get_wallet_balance: {e}"); return None

    def get_market_price(self, symbol=TRADING_SYMBOL):
        if not self.session: logger.error("Bybit session not available for get_market_price."); return None
        try:
            r = self.session.get_tickers(category="spot", symbol=symbol)
            if r and r.get('retCode') == 0:
                lst = r.get('result', {}).get('list', [])
                if lst: return Decimal(lst[0].get('lastPrice'))
            logger.error(f"ByBit API Error get_market_price: {r.get('retMsg') if r else 'No response'}")
            return None
        except Exception as e: logger.error(f"Exception in get_market_price: {e}"); return None

    def place_market_order(self, symbol, side, qty_str, category="spot"):
        if not self.session:
            logger.error("Bybit session not available for place_market_order.")
            return None, "SESSION_NOT_AVAILABLE"

        order_params = {
            "category": category,
            "symbol": symbol,
            "side": side,  # 'Buy' or 'Sell'
            "orderType": "Market",
            "qty": str(qty_str),  # Ensure qty is a string
        }
        try:
            logger.info(f"Placing order with params: {order_params} (Testnet: {self.is_testnet})")
            response = self.session.place_order(**order_params)
            logger.debug(f"Raw ByBit place_order response: {response}")

            if response and response.get('retCode') == 0:
                order_id = response.get('result', {}).get('orderId')
                if order_id:
                    logger.info(f"Successfully placed {side} order for {qty_str} {symbol}. Order ID: {order_id}")
                    return order_id, str(response) # Return orderId and raw response for logging
                else:
                    logger.error(f"Order placement succeeded (retCode 0) but no orderId returned. Response: {response}")
                    return None, f"LIVE_ORDER_NO_ORDER_ID_RETCODE_0: {str(response)}"
            else:
                ret_code = response.get('retCode') if response else 'N/A'
                ret_msg = response.get('retMsg') if response else 'No response object'
                logger.error(f"Failed to place order. retCode: {ret_code}, retMsg: '{ret_msg}'. Params: {order_params}")
                return None, f"LIVE_ORDER_FAILED_API: retCode={ret_code}, retMsg='{ret_msg}', Response: {str(response)}"

        except Exception as e:
            logger.error(f"Exception during place_market_order: {e}. Params: {order_params}")
            return None, f"LIVE_ORDER_EXCEPTION: {str(e)}"

class OllamaAdvisor:
    def __init__(self, endpoint, model_name): self.endpoint, self.model_name = endpoint, model_name; logger.info(f"OllamaAdvisor init: {model_name} at {endpoint}")
    def construct_prompt(self, symbol, market_price, balance_usdt, trading_goal, current_base_balance=None, recent_prices_str="N/A", qp_summary="N/A"):
        # Ensure balances are formatted to a reasonable number of decimal places for the prompt
        market_price_str = f"{market_price:.2f}"
        balance_usdt_str = f"{balance_usdt:.2f}"
        current_base_balance_str = f"{current_base_balance:.8f}" if current_base_balance is not None else "0.0"
        base_curr = symbol.replace("USDT", "") # Extract base currency (e.g., BTC)

        prompt = (
            f"Given the current market conditions for {symbol}: Current Price=${market_price_str} USD. "
            f"My current portfolio: {balance_usdt_str} USDT, {current_base_balance_str} {base_curr}. "
            f"My trading goal is: '{trading_goal}'. "
            f"Recent relative price changes (last {RL_ENV_WINDOW_SIZE} periods, if available, normalized to mean): {recent_prices_str}. "
            f"Optional quantum optimizer suggestion: {qp_summary}. "
            f"Based *only* on the information provided, should I BUY, SELL, or HOLD {base_curr}? "
            f"If BUY or SELL, what quantity of {base_curr} would you suggest? "
            f"Respond with ONLY a JSON object in the format: "
            f"{{\"action\": \"BUY|SELL|HOLD\", \"quantity\": \"<float_string_or_null>\", \"reasoning\": \"<brief_reasoning>\"}}."
        )
        return prompt

    def get_trading_advice(self, prompt):
        agent_raw_response = "N/A"
        try:
            logger.debug(f"OllamaAdvisor: Getting advice for prompt (len {len(prompt)})...")
            # logger.debug(f"Ollama Prompt: {prompt}") # Uncomment for deep debugging of prompt

            if "dummy_ollama" in self.endpoint: # Allow testing RL without real Ollama
                actions = ["BUY", "SELL", "HOLD"]
                action = actions[int(time.time()) % 3]
                qty = Decimal("0.0001") if action == "BUY" else (Decimal("0.00005") if action == "SELL" else None)
                dummy_response = {"action": action, "quantity": str(qty) if qty else None, "reasoning": "Dummy response"}
                agent_raw_response = json.dumps(dummy_response)
                return action, qty, agent_raw_response

            payload = {"model": self.model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 150}} # Added num_predict
            r = requests.post(self.endpoint, json=payload, timeout=45)  # Increased timeout
            r.raise_for_status()
            data = r.json()
            agent_raw_response = data.get('response','').strip()

            logger.debug(f"Ollama raw response: {agent_raw_response}")

            # Attempt to find JSON within the response if the model is verbose
            try:
                json_start_index = agent_raw_response.find('{')
                json_end_index = agent_raw_response.rfind('}') + 1
                if json_start_index != -1 and json_end_index != -1:
                    json_str = agent_raw_response[json_start_index:json_end_index]
                    parsed_response = json.loads(json_str)
                else: # Fallback if no clear JSON object is found
                    logger.warning(f"Ollama: No JSON object found in response: {agent_raw_response}")
                    return "HOLD", None, f"ERR_OLLAMA_NO_JSON: {agent_raw_response}"
            except json.JSONDecodeError as je:
                logger.error(f"Ollama JSON parsing error: {je} from response: {agent_raw_response}")
                return "HOLD", None, f"ERR_OLLAMA_JSON_DECODE: {agent_raw_response}"

            action = parsed_response.get("action", "HOLD").upper()
            quantity_str = parsed_response.get("quantity")

            # Validate action
            if action not in ["BUY", "SELL", "HOLD"]:
                logger.warning(f"Ollama: Invalid action '{action}' received. Defaulting to HOLD.")
                action = "HOLD"

            if action == "HOLD":
                return "HOLD", None, agent_raw_response

            if quantity_str is None or quantity_str.lower() == 'null':
                 logger.warning(f"Ollama: Action {action} received but quantity is null. Defaulting to HOLD.")
                 return "HOLD", None, agent_raw_response

            try:
                quantity = Decimal(quantity_str)
                if quantity <= Decimal("0"):
                    logger.warning(f"Ollama: Action {action} with non-positive quantity {quantity}. Defaulting to HOLD.")
                    return "HOLD", None, agent_raw_response
                return action, quantity, agent_raw_response
            except Exception as e:
                logger.error(f"Ollama: Error converting quantity '{quantity_str}' to Decimal: {e}. Defaulting to HOLD.")
                return "HOLD", None, agent_raw_response

        except requests.exceptions.RequestException as re:
            logger.error(f"Ollama request exception: {re}")
            return "HOLD", None, f"ERR_OLLAMA_REQUEST_FAILED: {re}"
        except Exception as e:
            logger.error(f"Ollama general exception: {e} processing response: {agent_raw_response}")
            return "HOLD", None, f"ERR_OLLAMA_GENERAL: {e}"


def get_current_observation_for_rl(current_price, price_history_window, usdt_balance, base_balance, initial_usdt_balance, window_size):
    # Ensure price_history_window has enough data, pad if necessary
    if len(price_history_window) < window_size:
        # Use current_price for padding if history is shorter than window
        padding_price = price_history_window[0] if price_history_window else current_price
        padding = [padding_price] * (window_size - len(price_history_window))
        norm_price_window_data = padding + price_history_window
    else:
        # Take the most recent 'window_size' elements
        norm_price_window_data = price_history_window[-window_size:]
    
    mean_val = sum(norm_price_window_data)/len(norm_price_window_data) if norm_price_window_data else Decimal('1') # Avoid division by zero
    norm_prices = [(p/mean_val)-Decimal('1') for p in norm_price_window_data]
    
    norm_usdt = usdt_balance / initial_usdt_balance if initial_usdt_balance > 0 else Decimal('0')
    
    # Use current_price if history is empty for typical_price calculation
    typ_price = price_history_window[0] if price_history_window else current_price 
    max_base = initial_usdt_balance / typ_price if typ_price > 0 else Decimal('1') # Avoid division by zero
    norm_base = base_balance / max_base if max_base > 0 else Decimal('0')
    
    return np.array([float(p) for p in norm_prices] + [float(norm_usdt), float(norm_base)], dtype=np.float32)


def run_trading_cycle(trader, portfolio_state, cycle_id, config, agent_type='ollama', ollama_advisor=None, rl_agent=None, price_hist_for_rl=None, quantum_opt=None):
    logger.info(f"--- Cycle {cycle_id} (Agent: {agent_type.upper()}) ---")
    current_price = trader.get_market_price(symbol=TRADING_SYMBOL)
    
    if current_price is None: 
        logger.error("Failed to get market price. Skipping cycle.")
        # Maintain price history by appending last known or a default if history is empty
        if price_hist_for_rl is not None: 
            price_hist_for_rl.append(price_hist_for_rl[-1] if price_hist_for_rl else Decimal("0.0")) # Append last known or 0
        return portfolio_state 
    
    if price_hist_for_rl is not None:
        price_hist_for_rl.append(current_price)
        # Keep price_hist_for_rl from growing indefinitely, ensuring it has enough for window + current
        while len(price_hist_for_rl) > RL_ENV_WINDOW_SIZE + 20: # Keep a bit more than window
             price_hist_for_rl.pop(0)

    usdt_bal, base_bal = portfolio_state[QUOTE_CURRENCY], portfolio_state[BASE_CURRENCY]
    logger.info(f"Portfolio: {usdt_bal:.2f} {QUOTE_CURRENCY}, {base_bal:.8f} {BASE_CURRENCY} | Price: ${current_price:.2f}")
    pf_usdt_before, pf_base_before = usdt_bal, base_bal
    q_summary = "N/A"
    if quantum_opt: q_summary = str(quantum_opt.generate_parameters()) # Simplified for this context

    agent_act, agent_qty, agent_raw = "HOLD", None, "N/A" # Defaults

    if agent_type == 'ollama':
        if ollama_advisor:
            recent_prices_summary = "N/A"
            if price_hist_for_rl and len(price_hist_for_rl) >= RL_ENV_WINDOW_SIZE:
                # Use the same normalization idea as for RL agent, but as a string
                # Take the most recent 'RL_ENV_WINDOW_SIZE' elements for summary
                relevant_history = price_hist_for_rl[-(RL_ENV_WINDOW_SIZE + 1):-1] # prices leading up to current
                if not relevant_history: relevant_history = [current_price] * RL_ENV_WINDOW_SIZE # fallback

                mean_val = sum(relevant_history)/len(relevant_history) if relevant_history else Decimal('1')
                norm_prices_for_ollama = [(p/mean_val)-Decimal('1') for p in relevant_history]
                recent_prices_summary = ", ".join([f"{p:.4f}" for p in norm_prices_for_ollama])

            prompt = ollama_advisor.construct_prompt(
                TRADING_SYMBOL, current_price, usdt_bal,
                config.get('trading_goal',"undefined goal"), base_bal,
                recent_prices_str=recent_prices_summary, qp_summary=q_summary
            )
            agent_act, agent_qty, agent_raw = ollama_advisor.get_trading_advice(prompt)
        else: logger.error("Ollama agent selected, but no advisor instance provided. Holding.")
    
    elif agent_type == 'rl':
        if rl_agent and price_hist_for_rl and len(price_hist_for_rl) >= RL_ENV_WINDOW_SIZE:
            # Use prices *leading up to* current_price for decision making state
            rl_decision_price_history = price_hist_for_rl[-(RL_ENV_WINDOW_SIZE+1):-1] if len(price_hist_for_rl) > RL_ENV_WINDOW_SIZE else price_hist_for_rl[:-1]
            if not rl_decision_price_history: # Edge case: history too short even after check
                logger.warning("RL: Not enough price history for decision observation, using current price for all window slots.")
                rl_decision_price_history = [current_price] * RL_ENV_WINDOW_SIZE


            initial_usdt_for_norm = Decimal(config.get('initial_usdt_balance','100.0')) # Fallback if not in config
            rl_obs = get_current_observation_for_rl(
                current_price, # Pass current price for context if needed by get_current_observation_for_rl
                rl_decision_price_history, 
                usdt_bal, base_bal, 
                initial_usdt_for_norm, 
                RL_ENV_WINDOW_SIZE)
            
            d_state = discretize_state(rl_obs, RL_ENV_WINDOW_SIZE)
            rl_action_idx = rl_agent.choose_action(d_state, is_training=False) # RL agent is for inference here
            agent_act = ['HOLD', 'BUY', 'SELL'][rl_action_idx]
            agent_raw = f"RL_Act_Idx:{rl_action_idx}, State:{d_state}"
            # RL agent does not suggest quantity; this is handled by risk management rules.
        else: 
            logger.error("RL agent selected, but agent instance or sufficient price history not available. Holding.")

    logger.info(f"Agent ({agent_type}) -> Decision: {agent_act}, Suggested Qty (Ollama): {agent_qty if agent_qty else 'N/A'}")
    
    # --- Execution based on agent's decision ---
    final_executed_action, final_executed_qty, trade_value_usdt = agent_act, Decimal("0"), Decimal("0")
    order_id, status, order_response_log_msg = "N/A", "PENDING_EXECUTION", "N/A"

    trade_percent = Decimal(config.get('default_trade_percentage', str(DEFAULT_USDT_TRADE_PERCENTAGE)))
    # Define precision for quantity string formatting based on symbol or general use
    # For BTCUSDT, Bybit might require up to 8 decimal places for BTC quantity.
    # Let's use a general precision that's likely safe for BTC.
    qty_precision = Decimal('0.00000001')


    if agent_act == "BUY":
        if usdt_bal >= MIN_USDT_ORDER_VALUE:
            qty_to_buy_calc = (usdt_bal * trade_percent / current_price)
            if agent_type == 'ollama' and agent_qty and agent_qty > 0:
                 qty_to_buy_calc = min(qty_to_buy_calc, agent_qty)
            
            # Ensure quantity is quantized before any other checks or string conversion
            qty_to_buy = qty_to_buy_calc.quantize(qty_precision, ROUND_DOWN)

            if qty_to_buy >= MIN_BTC_ORDER_QTY:
                val_usdt = (qty_to_buy * current_price).quantize(Decimal('0.01'), ROUND_UP)
                if val_usdt <= usdt_bal:
                    final_executed_qty_str = str(qty_to_buy) # Format to string *after* all Decimal calculations
                    actual_order_id, order_response_log_msg = trader.place_market_order(TRADING_SYMBOL, "Buy", final_executed_qty_str)
                    order_id = actual_order_id # For logging in log_trade_action
                    if actual_order_id:
                        status = "LIVE_BUY_SUCCESS"
                        trade_value_usdt = val_usdt # Log the actual trade value
                        final_executed_qty = qty_to_buy # Store Decimal for portfolio update
                        portfolio_state[QUOTE_CURRENCY] -= val_usdt
                        portfolio_state[BASE_CURRENCY] += final_executed_qty
                    else:
                        status = f"LIVE_BUY_FAILED ({order_response_log_msg})"
                        final_executed_action = "HOLD" # Mark as HOLD if order failed
                else: status = "REJECTED_BUY_AFFORDABILITY"; final_executed_action = "HOLD"
            else: status = "REJECTED_BUY_QTY_TOO_LOW"; final_executed_action = "HOLD"
        else: status = "REJECTED_BUY_LOW_USDT"; final_executed_action = "HOLD"

    elif agent_act == "SELL":
        if base_bal >= MIN_BTC_ORDER_QTY:
            qty_to_sell_calc = (base_bal * trade_percent)
            if agent_type == 'ollama' and agent_qty and agent_qty > 0:
                qty_to_sell_calc = min(qty_to_sell_calc, agent_qty)

            # Ensure quantity is quantized and also capped by available balance before other checks
            qty_to_sell = qty_to_sell_calc.quantize(qty_precision, ROUND_DOWN)
            qty_to_sell = min(qty_to_sell, base_bal)


            if qty_to_sell >= MIN_BTC_ORDER_QTY:
                val_usdt = (qty_to_sell * current_price).quantize(Decimal('0.01'), ROUND_DOWN)
                if val_usdt >= MIN_USDT_ORDER_VALUE:
                    final_executed_qty_str = str(qty_to_sell) # Format to string *after* all Decimal calculations
                    actual_order_id, order_response_log_msg = trader.place_market_order(TRADING_SYMBOL, "Sell", final_executed_qty_str)
                    order_id = actual_order_id # For logging
                    if actual_order_id:
                        status = "LIVE_SELL_SUCCESS"
                        trade_value_usdt = val_usdt # Log the actual trade value
                        final_executed_qty = qty_to_sell # Store Decimal for portfolio update
                        portfolio_state[QUOTE_CURRENCY] += val_usdt
                        portfolio_state[BASE_CURRENCY] -= final_executed_qty
                    else:
                        status = f"LIVE_SELL_FAILED ({order_response_log_msg})"
                        final_executed_action = "HOLD" # Mark as HOLD if order failed
                else: status = "REJECTED_SELL_VALUE_TOO_LOW"; final_executed_action = "HOLD"
            else: status = "REJECTED_SELL_QTY_TOO_LOW"; final_executed_action = "HOLD"
        else: status = "REJECTED_SELL_LOW_BASE"; final_executed_action = "HOLD"
    
    if agent_act == "HOLD" or final_executed_action == "HOLD":
        final_executed_action = "HOLD"
        status = "DECIDED_HOLD" if status == "PENDING_EXECUTION" else status
        final_executed_qty = Decimal("0")
        trade_value_usdt = Decimal("0")
        # Ensure order_id remains "N/A" or any specific failure message if an attempt was made
        if status not in ["LIVE_BUY_SUCCESS", "LIVE_SELL_SUCCESS"] and "FAILED" not in status and "EXCEPTION" not in status:
             order_id = "N/A"
    
    # Log the raw response from order placement if it's not a success and contains info
    if "FAILED" in status or "EXCEPTION" in status:
        logger.info(f"Order Response Log for failed/exception trade: {order_response_log_msg}")

    log_data = {
        "timestamp": datetime.now().isoformat(), "symbol": TRADING_SYMBOL, "action": final_executed_action,
        "price": str(current_price.quantize(Decimal('0.01'))),
        "quantity_base": str(final_executed_qty.quantize(qty_precision)),
        "value_quote": str(trade_value_usdt.quantize(Decimal('0.01'))),
        "agent_type": agent_type, "agent_raw_advice": agent_raw, "agent_action": agent_act,
        "agent_qty_suggestion": str(agent_qty.quantize(qty_precision) if agent_qty else "N/A"),
        "quantum_summary": q_summary, "order_id": order_id if order_id else "N/A", "status": status,
        "portfolio_usdt_before": str(pf_usdt_before.quantize(Decimal('0.01'))),
        "portfolio_base_before": str(pf_base_before.quantize(qty_precision)),
        "portfolio_usdt_after": str(portfolio_state[QUOTE_CURRENCY].quantize(Decimal('0.01'))), 
        "portfolio_base_after": str(portfolio_state[BASE_CURRENCY].quantize(qty_precision)), # Corrected precision here
        "cycle_id": str(cycle_id)
    }
    log_trade_action(log_data) # Pass the prepared dictionary

    save_portfolio_state(portfolio_state)
    logger.info(f"End Cycle {cycle_id}: Executed: {final_executed_action}, Qty: {final_executed_qty.quantize(qty_precision)}, Status: {status}. Portfolio: {portfolio_state[QUOTE_CURRENCY]:.2f} {QUOTE_CURRENCY}, {portfolio_state[BASE_CURRENCY]:.8f} {BASE_CURRENCY}")
    return portfolio_state

def main():
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    parser.add_argument('--setup-api', action='store_true', help="Setup API keys and initial settings")
    parser.add_argument('--bybit-key', help="ByBit API Key")
    parser.add_argument('--bybit-secret', help="ByBit API Secret")
    parser.add_argument('--testnet', action='store_true', default=None, help="Use ByBit testnet (default if not live)")
    parser.add_argument('--live', action='store_false', dest='testnet', help="Use ByBit livenet (EXTREME CAUTION)")
    parser.add_argument('--ollama-endpoint', help="Ollama API endpoint")
    parser.add_argument('--ollama-model', help="Ollama model name")
    parser.add_argument('--trading-goal', type=str, help="Define the trading goal for Ollama prompt")
    parser.add_argument('--initial-usdt', type=str, help="Set initial USDT for portfolio simulation")
    parser.add_argument('--default-trade-percentage', type=str, help="Default trade percentage for BUY/SELL (e.g., '0.03' for 3%)")

    parser.add_argument('--get-real-balance', action='store_true', help="Get REAL wallet balance from ByBit")
    parser.add_argument('--get-price', action='store_true', help="Get current market price from ByBit API")
    
    parser.add_argument('--run-trader', action='store_true', help="Run conceptual trading cycles")
    parser.add_argument('--cycles', type=int, default=1, help="Number of trading cycles for --run-trader")
    parser.add_argument('--use-quantum', action='store_true', help="Enable Quantum Optimizer (dummy version)")
    parser.add_argument('--quantum-qubits', type=int, default=5, help="Qubits for Quantum Optimizer")
    
    parser.add_argument('--view-log', action='store_true', help="View the trade log CSV")
    parser.add_argument('--view-portfolio', action='store_true', help="View current conceptual portfolio state")
    parser.add_argument('--reset-portfolio', action='store_true', help="Reset portfolio to initial state")

    parser.add_argument('--agent-type', type=str, default='ollama', choices=['ollama', 'rl'], help="Type of agent to use for trading decisions")
    parser.add_argument('--rl-qtable-path', type=str, default=DEFAULT_RL_Q_TABLE_PATH, help="Path to the RL agent's Q-table file")
    
    args = parser.parse_args(); config = load_config(); init_trade_log()

    if args.testnet is None: use_testnet = config.get('bybit_testnet', True)
    else: use_testnet = args.testnet
    config['bybit_testnet'] = use_testnet
    if not use_testnet: logger.warning("LIVENET MODE ENABLED. REAL TRADES WILL BE PLACED. EXTREME CAUTION ADVISED.")

    if args.setup_api:
        config['bybit_api_key'] = args.bybit_key or input("ByBit Key: ")
        config['bybit_api_secret'] = args.bybit_secret or input("ByBit Secret: ")
        config['ollama_endpoint'] = args.ollama_endpoint or config.get('ollama_endpoint', DEFAULT_OLLAMA_ENDPOINT)
        config['ollama_model'] = args.ollama_model or config.get('ollama_model', DEFAULT_OLLAMA_MODEL)
        config['trading_goal'] = args.trading_goal or config.get('trading_goal', "Maximize profit while minimizing losses.")
        config['initial_usdt_balance'] = args.initial_usdt or config.get('initial_usdt_balance', "3.0")
        config['default_trade_percentage'] = args.default_trade_percentage or config.get('default_trade_percentage', str(DEFAULT_USDT_TRADE_PERCENTAGE))
        config['rl_qtable_path'] = args.rl_qtable_path or config.get('rl_qtable_path', DEFAULT_RL_Q_TABLE_PATH)
        save_config(config); logger.info("Config updated."); 
        if input("Reset portfolio to new initial balance? (y/N): ").lower() == 'y':
            save_portfolio_state({QUOTE_CURRENCY: Decimal(config['initial_usdt_balance']), BASE_CURRENCY: Decimal("0.0")})
        return
    
    # View/Reset Handlers (simplified for brevity, assume they exist and work as before)
    if args.view_log: print("Log viewing TBD"); return
    if args.view_portfolio: print("Portfolio viewing TBD"); return
    if args.reset_portfolio: save_portfolio_state({QUOTE_CURRENCY: Decimal(config.get('initial_usdt_balance',"3.0")), BASE_CURRENCY:Decimal("0.0")}); logger.info("Portfolio reset.");return

    api_key, api_secret = config.get('bybit_api_key'), config.get('bybit_api_secret')
    trader = None
    if api_key and api_secret:
        try: trader = ByBitTrader(api_key, api_secret, testnet=use_testnet)
        except Exception as e: logger.error(f"Trader init error: {e}")
    
    if args.get_real_balance: 
        if trader: bal = trader.get_wallet_balance(); logger.info(f"Real Balance: {bal}") 
        else: logger.error("No trader for real balance.")
        return
    if args.get_price: 
        if trader: price = trader.get_market_price(); logger.info(f"Price: {price}")
        else: logger.error("No trader for price.")
        return

    ollama_adv, rl_agent_inst = None, None # Initialize to None

    if args.agent_type == 'ollama' or (args.run_trader and args.agent_type == 'ollama'):
        ollama_ep = args.ollama_endpoint or config.get('ollama_endpoint', DEFAULT_OLLAMA_ENDPOINT)
        if not ollama_ep or "dummy_ollama" in ollama_ep: # Use dummy if not properly set or explicitly dummy
            logger.warning("Using dummy Ollama endpoint for Ollama agent.")
            ollama_ep = "dummy_ollama_endpoint" 
        ollama_adv = OllamaAdvisor(ollama_ep, args.ollama_model or config.get('ollama_model', DEFAULT_OLLAMA_MODEL))

    if args.agent_type == 'rl' or (args.run_trader and args.agent_type == 'rl'):
        rl_agent_inst = QLearningAgent(action_space_n=3) # 3 actions: HOLD, BUY, SELL
        q_path = args.rl_qtable_path or config.get('rl_qtable_path', DEFAULT_RL_Q_TABLE_PATH)
        rl_agent_inst.load_q_table(q_path) # Load Q-table
        if not rl_agent_inst.q_table: # q_table is a defaultdict, so check if it's empty
            logger.error(f"CRITICAL: RL Q-table at '{q_path}' is empty or could not be loaded. "
                         f"The RL agent will not function correctly and will make random decisions. "
                         f"Please ensure the Q-table is generated by training 'rl_agent.py' or 'application.py'.")
            # Depending on operational requirements, you might want to prevent trading:
            # if not use_testnet: # Example: prevent live trading with an untrained agent
            #     logger.critical("EXITING due to empty Q-table in a non-testnet environment.")
            #     return
        elif q_path == DEFAULT_RL_Q_TABLE_PATH and len(rl_agent_inst.q_table) < 10: # Arbitrary small number
             logger.warning(f"RL Q-table at default path '{q_path}' seems very small (entries: {len(rl_agent_inst.q_table)}). "
                            f"Ensure it's adequately trained.")
        else:
             logger.info(f"RL Q-table loaded from '{q_path}' with {len(rl_agent_inst.q_table)} entries.")


    if args.run_trader:
        if not trader:
            logger.warning("ByBit trader not available (check API keys or connectivity). Using OFFLINE mock trader for simulation.")
            class OfflineTraderMock:
                def __init__(self, testnet=True): self.is_testnet=testnet; self.mock_price=Decimal(config.get('initial_mock_price',"20000.00")) # Make mock price configurable
                def get_market_price(self,symbol=""): self.mock_price*=(Decimal('1')+Decimal(np.random.normal(0,0.005))); return self.mock_price.quantize(Decimal('0.01'))
                def place_market_order(self,symbol,side,qty,category=""): return f"OFFLINE_STUB_{int(time.time())}"
            trader = OfflineTraderMock(testnet=use_testnet)

        active_agent_type = args.agent_type
        if active_agent_type == 'ollama' and not ollama_adv: logger.error("Ollama agent selected but not initialized."); return
        if active_agent_type == 'rl' and not rl_agent_inst: logger.error("RL agent selected but not initialized."); return
            
        quantum_instance = QuantumInspiredOptimizer(n_qubits=args.quantum_qubits) if args.use_quantum else None
        current_portfolio_state = load_portfolio_state()
        
        # Initialize price history for RL agent observation window
        price_history_for_rl = []
        initial_price_points = RL_ENV_WINDOW_SIZE + 5 # Get a bit more than window to start
        logger.info(f"Initializing price history with {initial_price_points} points for RL Agent...")
        for _ in range(initial_price_points):
            # Use trader.get_market_price() which handles offline mock correctly
            hist_price = trader.get_market_price() 
            if hist_price is None: # Should only happen if real API fails and mock isn't used
                logger.warning("Failed to get initial price for history, using a default.")
                hist_price = Decimal(config.get('initial_mock_price', "20000.00")) # Default if API fails
            price_history_for_rl.append(hist_price)
            if trader.__class__.__name__ == "OfflineTraderMock": time.sleep(0.01) # Small delay for mock price changes

        for i in range(args.cycles):
            current_portfolio_state = run_trading_cycle(trader, current_portfolio_state, i+1, config, 
                                                        active_agent_type, ollama_adv, rl_agent_inst, 
                                                        price_history_for_rl, quantum_instance)
            if i < args.cycles -1 : 
                logger.info(f"Cycle {i+1} finished. Waiting 1 second before next cycle...")
                time.sleep(1) 
        return

    # Fallback to help if no action args given
    action_args_given = any([
        args.setup_api, args.get_real_balance, args.get_price, args.run_trader,
        args.view_log, args.view_portfolio, args.reset_portfolio
    ])
    if not action_args_given:
        parser.print_help()

if __name__ == "__main__":
    main()
