# Trading Bot CLI (Conceptual & Simulated)

This project implements a command-line interface (CLI) for a conceptual trading bot.
It integrates with ByBit (for market data and STUBBED order placement), Ollama (for AI-driven trading advice),
and a Qiskit-based quantum-inspired optimizer (for parameter suggestion).

**IMPORTANT: This bot is currently for SIMULATION AND EDUCATIONAL PURPOSES ONLY.**
- Order placement with ByBit is STUBBED. No real trades are made.
- The portfolio is tracked conceptually in a local JSON file.
- The effectiveness of Ollama and the quantum-inspired optimizer for actual trading is highly speculative and requires extensive research and refinement.

## Features

- Basic CLI for interaction.
- Configuration management for API keys and endpoints (`config.json`).
- ByBit API integration (testnet/livenet) for fetching market data and (stubbed) orders.
- Ollama integration for generating trading advice based on market data and goals.
- Quantum-inspired parameter generation using Qiskit.
- Core trading loop combining these elements with basic risk management.
- CSV logging of all (conceptual) trade actions (`trade_log.csv`).
- JSON-based conceptual portfolio tracking (`portfolio_state.json`).

## Setup and Usage

1.  **Clone the repository.**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Ollama:**
    - Ensure you have Ollama installed and running.
    - Pull a model that you want to use, e.g., `ollama pull llama3`. The default in the script is 'llama3'.
5.  **Configure the bot:**
    - Run the setup command to store your ByBit API keys (preferably Testnet keys), Ollama settings, and initial portfolio balance.
    ```bash
    python trading_cli.py --setup-api
    ```
    - You will be prompted for your ByBit API key and secret.
    - You can also set your initial USDT balance for simulation (e.g., `--initial-usdt 3.0`).

6.  **Explore commands:**
    ```bash
    python trading_cli.py --help
    ```
    - **Run trading simulation:**
      ```bash
      # Run 1 cycle, without quantum optimizer, using faked price if Bybit keys not set
      python trading_cli.py --run-trader
      # Run 3 cycles, with quantum optimizer (needs Bybit keys for real price, else faked)
      python trading_cli.py --run-trader --cycles 3 --use-quantum
      ```
    - **View logs and portfolio:**
      ```bash
      python trading_cli.py --view-log
      python trading_cli.py --view-portfolio
      python trading_cli.py --reset-portfolio
      ```
    - **Fetch real data (requires valid API keys):**
      ```bash
      python trading_cli.py --get-real-balance
      python trading_cli.py --get-price
      ```

## Iteration and Refinement

This script is a starting point. Key areas for your own iteration include:
-   **Ollama Prompt Engineering:** Modify `OllamaAdvisor.construct_prompt` to improve LLM guidance.
-   **Risk Management:** Adjust `DEFAULT_USDT_TRADE_PERCENTAGE` and other parameters in `run_trading_cycle`. Implement more sophisticated rules (e.g., stop-loss, take-profit - currently not implemented).
-   **Quantum Optimizer:** Experiment with `QuantumInspiredOptimizer` circuit design and how its output is interpreted and used in the prompt/logic.
-   **Live Trading:** To move towards live trading (EXTREME CAUTION):
    -   Thoroughly test on ByBit Testnet.
    -   Implement actual order placement in `ByBitTrader.place_market_order`.
    -   Develop robust error handling and state management for live operations.
    -   Integrate comprehensive real-time monitoring and alerting.

## Packaging (Conceptual)

-   **PyInstaller:** To create a standalone executable:
    1.  `pip install pyinstaller`
    2.  `pyinstaller --onefile --name trading_bot trading_cli.py`
    (This might require tweaks for dependencies like Qiskit, especially data files it might use).
-   **Docker:** See the example `Dockerfile` provided.

## Monitoring a Live Bot (Conceptual)

If this bot were to be run live, consider:
-   **Log Management:** Centralized logging (ELK stack, Splunk, cloud services).
-   **Metrics & Alerting:** Prometheus/Grafana for system/app metrics. Alerts for errors, large losses, API failures (PagerDuty, etc.).
-   **Process Supervision:** Tools like `systemd` (Linux) or Windows Task Scheduler to ensure the bot runs continuously and restarts on failure.
-   **Health Checks:** A simple endpoint or mechanism for external systems to verify the bot is operational.
-   **Dashboarding:** Visualizing P&L, trade history, and portfolio metrics.

## Disclaimer

Trading cryptocurrencies is highly risky. This software is provided for educational and simulation purposes only. The authors and contributors are not responsible for any financial losses incurred. USE AT YOUR OWN RISK.
