# Ollama & RL Powered ByBit Trading Bot

## Overview

This Python-based trading bot is designed for automated trading on the ByBit cryptocurrency exchange. It integrates with ByBit for live or testnet trading, and leverages Large Language Models (LLMs) via Ollama and a Reinforcement Learning (RL) agent for generating trading signals.

**CRITICAL WARNING: Trading cryptocurrencies involves significant risk of financial loss. This software is experimental. Always start by using the Testnet environment and, if you choose to trade live, only use capital you can afford to lose. The authors are not responsible for any financial losses incurred by using this software.**

## Features

*   **Command-Line Interface (CLI):** All interactions with the bot are through `trading_cli.py`.
*   **Configuration:** Bot settings, API keys, and agent parameters are managed via `config.json`.
*   **ByBit API Integration:**
    *   Fetches real-time market data (prices, balances).
    *   Places **live market orders** on ByBit spot market (e.g., BTCUSDT).
    *   Supports both ByBit Mainnet and Testnet environments.
*   **Ollama Integration:**
    *   Connects to a local Ollama instance to get trading advice from LLMs.
    *   Supports configurable models (e.g., `llama3.2`, `qwen2.5vl`).
    *   Uses a structured JSON-based prompt for more reliable responses.
*   **Reinforcement Learning (RL) Agent:**
    *   A Q-learning agent that can be trained to make trading decisions (HOLD, BUY, SELL).
    *   Training and Q-table generation are handled by `rl_agent.py` (for direct training) and `application.py` (for evaluating/running training loops).
    *   The trained Q-table (`q_table.pkl`) is used by `trading_cli.py` for RL-based trading.
*   **QuantumInspiredOptimizer (Placeholder):**
    *   Includes a class `QuantumInspiredOptimizer`.
    *   **This is an EXPERIMENTAL/PLACEHOLDER feature and is NOT currently used for actual trading decisions or any optimization.**
*   **Logging & Tracking:**
    *   Detailed trade actions, agent advice, and portfolio changes are logged to `trade_log.csv`.
    *   The conceptual portfolio state (simulated balances) is saved in `portfolio_state.json`.

## Setup and Usage

### 1. Clone Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
The `requirements.txt` file has been updated to include necessary packages like `pybit`, `requests`, and `numpy`.
```bash
pip install -r requirements.txt
```

### 4. Ollama Setup
*   Ensure Ollama is installed and running locally. Visit [ollama.com](https://ollama.com) for installation instructions.
*   Pull the desired LLM models. Examples:
    ```bash
    ollama pull llama3 # Default model
    ollama pull llama3.2
    ollama pull qwen2 # For qwen2.5vl - check exact model name on Ollama Hub
    ```

### 5. Initial Bot Configuration
Use the `--setup-api` command to configure the bot for the first time. This will create or update `config.json`.
```bash
python trading_cli.py --setup-api
```
You will be prompted for:
*   **ByBit API Key & Secret:**
    *   **IMPORTANT:** Start with **Testnet API keys**. You can get these from your ByBit account settings.
    *   You can choose to use Testnet (default) or Livenet.
*   **Ollama Endpoint URL:** (e.g., `http://localhost:11434/api/generate`)
*   **Ollama Model Name:** (e.g., `llama3`, `llama3.2`)
*   **Initial USDT Balance:** For portfolio simulation (e.g., `1000.0`).
*   **Trading Goal:** A text description for the Ollama agent's prompt (e.g., "Aggressively grow capital with moderate risk tolerance for BTCUSDT.").
*   **RL Q-Table Path:** Default is `q_table.pkl`.

### 6. Running the Bot

*   **View Help:**
    ```bash
    python trading_cli.py --help
    ```

*   **Testnet Trading (Recommended First):**
    *   Ensure `bybit_testnet: true` is set in your `config.json`. This is the default during setup.
    *   Run with Ollama agent:
        ```bash
        python trading_cli.py --run-trader --agent-type ollama --cycles 10
        ```
    *   Run with RL agent (ensure `q_table.pkl` exists or is trained):
        ```bash
        python trading_cli.py --run-trader --agent-type rl --cycles 10
        ```

*   **Livenet Trading (EXTREME CAUTION):**
    *   **WARNING:** Live trading involves real financial risk. Ensure you understand the bot's behavior on Testnet before proceeding.
    *   Modify `config.json` to set `bybit_testnet: false`. You can do this by re-running `--setup-api` and selecting 'no' for testnet, or by manually editing the file.
    *   The `--live` flag can be used as an additional explicit confirmation if desired, but the primary control is `config.json`.
        ```bash
        # Ensure bybit_testnet is false in config.json first!
        python trading_cli.py --run-trader --agent-type ollama --cycles 1 --live
        # Or, relying on config.json:
        # python trading_cli.py --run-trader --agent-type ollama --cycles 1
        ```

*   **Other Utility Commands:**
    *   Get current market price: `python trading_cli.py --get-price`
    *   Get ByBit wallet balance (real or testnet based on config): `python trading_cli.py --get-real-balance`
    *   View trade log: `python trading_cli.py --view-log` (Currently prints "Log viewing TBD")
    *   View portfolio state: `python trading_cli.py --view-portfolio` (Currently prints "Portfolio viewing TBD")
    *   Reset portfolio state: `python trading_cli.py --reset-portfolio`

### 7. Training the RL Agent
The Reinforcement Learning agent (`QLearningAgent`) is trained using historical or simulated price data. The training logic is primarily within `rl_agent.py` (which can be run directly for a sample training loop) and `application.py` (designed for more structured training runs and evaluations).

*   **Run `application.py` to perform training episodes and generate `q_table.pkl`:**
    ```bash
    python application.py
    ```
    This script uses constants defined within it (like `PRICE_DATA_LENGTH`, `PRICE_START_PRICE`) to generate dummy price data for training the `TradingEnv` and then trains the `QLearningAgent`, saving the resulting Q-table.

## Agent Details

### Ollama Agent
*   The Ollama agent (`OllamaAdvisor`) constructs a detailed prompt for the selected LLM.
*   The prompt includes current market price, portfolio balance, trading goal, recent price changes (normalized), and any (placeholder) quantum optimizer summary.
*   It specifically requests the LLM to respond with a JSON object:
    `{"action": "BUY|SELL|HOLD", "quantity": "<float_string_or_null>", "reasoning": "<brief_reasoning>"}`
*   This structured JSON response is parsed to extract the trading action and quantity.
*   The model used by Ollama (e.g., `llama3`, `llama3.2`, `qwen2`) is configurable via `config.json` or `--setup-api`.

### RL Agent
*   The Reinforcement Learning agent (`QLearningAgent`) uses Q-learning to determine actions.
*   It relies on a pre-trained Q-table, typically saved as `q_table.pkl`.
*   **Observation Space:** The agent observes a window of past normalized price changes and current normalized portfolio balances (USDT and base currency). The size of this window (`RL_ENV_WINDOW_SIZE`) must be consistent between training (`trading_env.py`, `rl_agent.py`, `application.py`) and execution (`trading_cli.py`).
*   **State Discretization:** Continuous observation values are discretized into bins to be used as states in the Q-table. The bin boundaries are defined in `rl_agent.py`.
*   **Actions:** HOLD, BUY, SELL.

## Important Considerations / Disclaimer

*   **RISK OF FINANCIAL LOSS:** Trading cryptocurrencies is highly speculative and carries a significant risk of loss. This software is provided "as is," without any warranty, express or implied.
*   **EXPERIMENTAL SOFTWARE:** This bot is an experimental project. It may contain bugs or behave unexpectedly.
*   **NO GUARANTEE OF PROFIT:** There is no guarantee that this bot will generate profits. Market conditions can change rapidly, and past performance is not indicative of future results.
*   **TESTNET FIRST:** Always use the Testnet environment extensively to understand the bot's behavior before considering live trading.
*   **START WITH SMALL CAPITAL:** If you decide to trade live, start with a very small amount of capital that you are fully prepared to lose.
*   **AUTHOR NOT RESPONSIBLE:** The author(s) of this software are not responsible for any financial losses or damages incurred from its use.

## License
This project is intended for educational and experimental purposes. Please use responsibly. (Assuming MIT License if a `LICENSE` file is present, otherwise, this is a placeholder statement).

Consider adding a `LICENSE` file (e.g., MIT License) to the repository.
