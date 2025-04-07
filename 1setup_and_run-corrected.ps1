# Ensure script runs with elevated privileges if needed
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Please run this script as Administrator for full functionality." -ForegroundColor Yellow
}

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$NC = "`e[0m"

Write-Host "${Green}Starting setup for BTC Quantum script in PowerShell...${NC}"

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion"
} catch {
    Write-Host "${Red}Python not found. Please install Python 3.10 or later and add it to PATH.${NC}"
    exit 1
}

# Install Miniconda if not present
$minicondaPath = "$env:USERPROFILE\Miniconda3"
if (-not (Test-Path $minicondaPath)) {
    Write-Host "Installing Miniconda..."
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "Miniconda3-latest.exe"
    Start-Process -FilePath "Miniconda3-latest.exe" -ArgumentList "/S /D=$minicondaPath" -Wait
    Remove-Item "Miniconda3-latest.exe"
} else {
    Write-Host "Miniconda already installed."
}

# Initialize Conda for this session
$condaScript = "$minicondaPath\Scripts\conda.exe"
& "$condaScript" "shell.powershell" "hook" | Out-String | Invoke-Expression

# Create and activate Conda environment
Write-Host "Creating and activating Conda environment 'eth_env'..."
conda create -n eth_env python=3.10 -y
conda activate eth_env

# Install Python dependencies
Write-Host "Installing Python dependencies..."
conda install -c conda-forge -y qiskit=1.4.1 qiskit-aer=0.14.2 qiskit-ibm-runtime=0.37.0 requests=2.32.3 numpy=2.0.2 scipy=1.13.1 sympy=1.13.3 base58=2.1.1 ecdsa=0.19.1
pip install python-dateutil==2.9.0.post0 stevedore==5.2.0 typing-extensions==4.12.2 symengine==0.13.0 dill==0.3.8 psutil==6.0.0

# Set environment variables
$env:IBM_QUANTUM_TOKEN = "334d70c5d25a2b538142a228c1c9b30d02496429469596a62c3e158198caf82536d63b8ce7722ee7befe404626dac86c7c7bea212ee4c535dbfc516bf1c3963f"
$env:COINAPI_KEY = "ec42825f-62f6-4c0f-99b6-d0dfcad2498d"
$env:DATABASE_PATH = "$PWD\database\11_13_2022\"

# Create main.py with corrected IQFT handling
Write-Host "Creating main.py..."
Set-Content -Path "main.py" -Value @"
import os
import time
import hashlib
import requests
import sqlite3
import base58
import ecdsa
import binascii
import multiprocessing
import logging
import sys
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.exceptions import QiskitError

# Script version
SCRIPT_VERSION = "1.2.1 - PowerShell Compatible (Fixed IQFT)"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load configuration from environment variables with Windows-friendly paths
IBM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN', '334d70c5d25a2b538142a228c1c9b30d02496429469596a62c3e158198caf82536d63b8ce7722ee7befe404626dac86c7c7bea212ee4c535dbfc516bf1c3963f')
COINAPI_KEY = os.getenv('COINAPI_KEY', 'ec42825f-62f6-4c0f-99b6-d0dfcad2498d')
DATABASE_PATH = os.getenv('DATABASE_PATH', r'database\11_13_2022\\')  # Windows path with trailing separator

# Ensure database directory exists
if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_PATH)
    logger.info(f"Created database directory: {DATABASE_PATH}")
DATABASE_AVAILABLE = os.path.exists(DATABASE_PATH)
DB_FILE = os.path.join(DATABASE_PATH, 'btc_addresses.db')

# Initialize IBM Quantum service
try:
    service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q/open/main', token=IBM_TOKEN)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=20)
    logger.info("IBM Quantum service initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize IBM Quantum service: {e}. Using AerSimulator only.")

# Brainwallet key generation
def generate_brainwallet_key(passphrase: str, tweak: int = 0) -> Tuple[bytes, str]:
    try:
        base_key = hashlib.sha256(passphrase.encode()).digest()
        tweak_bytes = tweak.to_bytes(32, 'big')
        tweaked_key = bytes(a ^ b for a, b in zip(base_key, tweak_bytes))
        sk = ecdsa.SigningKey.from_string(tweaked_key, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        public_key = b'\x04' + vk.to_string()
        sha256_hash = hashlib.sha256(public_key).digest()
        ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
        extended_hash = b'\x00' + ripemd160_hash
        checksum = hashlib.sha256(hashlib.sha256(extended_hash).digest()).digest()[:4]
        address = base58.b58encode(extended_hash + checksum).decode('utf-8')
        return tweaked_key, address
    except Exception as e:
        logger.error(f"Error generating brainwallet key for passphrase '{passphrase}': {e}")
        raise

# Convert private key to WIF
def private_key_to_wif(priv_key: bytes) -> str:
    try:
        extended_key = b'\x80' + priv_key
        checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
        return base58.b58encode(extended_key + checksum).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting private key to WIF: {e}")
        raise

# Plutus database check
def check_balance_database(address: str, substring_length: int = 8) -> Tuple[str, int]:
    if not DATABASE_AVAILABLE:
        return address, 0
    try:
        substring = address[-substring_length:]
        for filename in os.listdir(DATABASE_PATH):
            with open(os.path.join(DATABASE_PATH, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                if substring in content:
                    f.seek(0)
                    if address in f.read():
                        return address, 1
        return address, 0
    except Exception as e:
        logger.error(f"Error checking database for address {address}: {e}")
        return address, 0

# CoinAPI balance check with retry
def check_balance_coinapi(address: str, retries: int = 2) -> int:
    url = f"https://rest.coinapi.io/v1/balances/bitcoin/address/{address}"
    headers = {'X-CoinAPI-Key': COINAPI_KEY}
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=2)
            response.raise_for_status()
            data = response.json()
            return int(data.get('balance', 0) * 100000000)
        except requests.RequestException as e:
            logger.warning(f"CoinAPI attempt {attempt + 1} failed for {address}: {e}")
            if attempt == retries:
                return check_balance_blockchain(address)
            time.sleep(1)

# Blockchain.com fallback
def check_balance_blockchain(address: str) -> int:
    url = f"https://blockchain.info/q/addressbalance/{address}"
    try:
        response = requests.get(url, timeout=1)
        return int(response.text)
    except requests.RequestException as e:
        logger.warning(f"Blockchain.com check failed for {address}: {e}")
        return 0

# Batch balance checking
def check_balance_batch(addresses: List[str]) -> List[Tuple[str, int]]:
    try:
        results = list(map(check_balance_database, addresses))
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_addr = {executor.submit(check_balance_coinapi, addr): addr for addr, bal in results if bal > 0}
            verified = {}
            for future in future_to_addr:
                try:
                    verified[future_to_addr[future]] = future.result(timeout=5)
                except FuturesTimeoutError:
                    logger.warning(f"Timeout checking {future_to_addr[future]} with CoinAPI")
                    verified[future_to_addr[future]] = check_balance_blockchain(future_to_addr[future])
        return [(addr, verified.get(addr, bal if bal == 0 else check_balance_coinapi(addr))) for addr, bal in results]
    except Exception as e:
        logger.error(f"Error in batch balance check: {e}")
        return [(addr, 0) for addr in addresses]

# Local quantum sandbox
def local_quantum_sandbox(passphrase: str, n_bits: int) -> List[int]:
    try:
        qreg = QuantumRegister(n_bits, 'q')
        creg = ClassicalRegister(n_bits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        qc.h(range(n_bits))  # Superposition
        qc.cz(0, n_bits - 1)  # Simplified oracle
        qc.append(QFT(n_bits, do_swaps=False, inverse=True), range(n_bits))  # Inverse QFT
        qc.measure(range(n_bits), range(n_bits))
        
        simulator = AerSimulator(method='statevector')
        job = simulator.run(qc, shots=1000)
        result = job.result()
        if not result.success:
            raise QiskitError("Local quantum simulation failed")
        counts = result.get_counts()
        
        tweaks = [int(state, 2) for state, count in counts.items() if count / 1000 > 0.05]
        logger.info(f"Local sandbox for '{passphrase}': {len(tweaks)} tweaks generated")
        return tweaks[:50]  # Top 50 candidates
    except QiskitError as qe:
        logger.error(f"Qiskit error in local_quantum_sandbox: {qe}")
        return []
    except Exception as e:
        logger.error(f"Error in local_quantum_sandbox: {e}")
        return []

# IBM Quantum refinement
def ibm_quantum_refine(passphrase: str, n_bits: int, candidates: List[int]) -> List[int]:
    try:
        qreg = QuantumRegister(n_bits, 'q')
        creg = ClassicalRegister(n_bits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        qc.h(range(n_bits))
        qc.cz(0, n_bits - 1)
        qc.append(QFT(n_bits, do_swaps=False, inverse=True), range(n_bits))
        qc.measure(range(n_bits), range(n_bits))
        
        sampler = Sampler(mode=backend)
        job = sampler.run([qc], shots=1000)
        result = job.result()
        counts = result[0].data.c.get_counts()
        
        tweaks = [int(state, 2) for state, count in counts.items() if count / 1000 > 0.05]
        logger.info(f"IBM refine for '{passphrase}': {len(tweaks)} tweaks refined")
        return tweaks[:20]  # Top 20 candidates
    except QiskitError as qe:
        logger.error(f"Qiskit error in ibm_quantum_refine: {qe}")
        return candidates[:20]  # Fallback to local candidates
    except Exception as e:
        logger.error(f"Error in ibm_quantum_refine: {e}")
        return candidates[:20]

# Worker function
def worker(passphrases: List[str], local_bits: int, ibm_bits: int, checked_count: multiprocessing.Value):
    try:
        logger.info(f"Worker started with SQLite3 version: {sqlite3.sqlite_version}")
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')  # Enable Write-Ahead Logging
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS addresses
                     (private_key TEXT, wif_key TEXT, address TEXT, balance INTEGER, timestamp TEXT)''')
        local_count = 0
        
        for passphrase in passphrases:
            local_tweaks = local_quantum_sandbox(passphrase, local_bits)
            if not local_tweaks:
                continue
            ibm_tweaks = ibm_quantum_refine(passphrase, ibm_bits, local_tweaks)
            if not ibm_tweaks:
                ibm_tweaks = local_tweaks[:20]  # Fallback
            
            keypairs = [generate_brainwallet_key(passphrase, tweak) for tweak in ibm_tweaks]
            addresses = [kp[1] for kp in keypairs]
            priv_keys = [kp[0] for kp in keypairs]
            
            balance_results = check_balance_batch(addresses)
            for (addr, bal), priv_key in zip(balance_results, priv_keys):
                if bal > 0:
                    wif_key = private_key_to_wif(priv_key)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    c.execute("INSERT INTO addresses VALUES (?, ?, ?, ?, ?)",
                              (priv_key.hex(), wif_key, addr, bal, timestamp))
                    conn.commit()
                    logger.info(f"Found: {addr} | Balance: {bal / 100000000:.8f} BTC | WIF: {wif_key}")
            local_count += len(ibm_tweaks)
        
        with checked_count.get_lock():
            checked_count.value += local_count
    except Exception as e:
        logger.error(f"Worker error: {e}")
    finally:
        conn.close()

# Debug quantum circuit
def debug_quantum_circuit(n_bits: int = 20) -> None:
    try:
        qreg = QuantumRegister(n_bits, 'q')
        creg = ClassicalRegister(n_bits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        qc.h(range(n_bits))  # Apply Hadamard gates
        qc.cz(0, n_bits - 1)  # Apply controlled-Z gate
        qc.append(QFT(n_bits, do_swaps=False, inverse=True), range(n_bits))  # Apply inverse QFT
        qc.measure(range(n_bits), range(n_bits))  # Measure all qubits
        
        logger.info("Debug circuit:")
        logger.info(qc.draw(output='text'))
        
        simulator = AerSimulator(method='statevector')
        job = simulator.run(qc, shots=1000)
        result = job.result()
        if not result.success:
            raise QiskitError("Debug simulation failed")
        counts = result.get_counts()
        logger.info("Debug: Quantum circuit ran successfully. Sample counts:")
        logger.info(list(counts.items())[:5])
    except QiskitError as qe:
        logger.error(f"Qiskit error in debug_quantum_circuit: {qe}")
    except Exception as e:
        logger.error(f"Error in debug_quantum_circuit: {e}")

# Main function
def main():
    logger.info(f"Starting script version: {SCRIPT_VERSION}")
    passphrases = [
        "password", "bitcoin", "123456", "secret", "letmein",
        "qwerty", "admin", "blockchain", "crypto", "money",
        "pass123", "btc2023", "wallet", "secure", "freedom",
        "password123", "bitcoin2023", "123456789", "test", "hello",
        "abc123", "mywallet", "cash", "satoshi", "hodl",
        "admin123", "pass", "love", "root", "private",
        "password1", "bitcoin123", "qwerty123", "secure123", "wallet123"
    ]
    local_bits = 20
    ibm_bits = 20
    
    # Debug mode
    debug = True
    if debug:
        logger.info("Running in debug mode")
        logger.info(f"SQLite3 version: {sqlite3.sqlite_version}")
        debug_quantum_circuit(local_bits)
        return
    
    checked_count = multiprocessing.Value('i', 0)
    processes = []
    
    try:
        start_time = time.time()
        cpu_count = min(20, multiprocessing.cpu_count())
        chunk_size = max(1, len(passphrases) // cpu_count)
        
        for i in range(cpu_count):
            chunk = passphrases[i * chunk_size:(i + 1) * chunk_size]
            if chunk:
                p = multiprocessing.Process(target=worker, args=(chunk, local_bits, ibm_bits, checked_count))
                processes.append(p)
                p.start()
        
        while time.time() - start_time < 120:  # 2-minute runtime
            elapsed = time.time() - start_time
            with checked_count.get_lock():
                total_keys = checked_count.value
            logger.info(f"Elapsed: {elapsed:.1f}s | Checked: {total_keys} keys | Database: {DATABASE_AVAILABLE}")
            time.sleep(5)
        
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        for p in processes:
            if p.is_alive():
                p.join()
        logger.info("Execution completed")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Required for Windows/PowerShell
    main()
"@

# Run the script in debug mode
Write-Host "${Green}Running main.py in debug mode...${NC}"
python main.py

# Optional: Run in full mode (uncomment to enable)
# Write-Host "Editing main.py to disable debug mode..."
# (Get-Content -Path "main.py") -replace 'debug = True', 'debug = False' | Set-Content -Path "main.py"
# Write-Host "${Green}Running main.py in full mode...${NC}"
# python main.py

Write-Host "${Green}Setup and execution complete!${NC}"
Write-Host "To reactivate the environment later, run: conda activate eth_env"
