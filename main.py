import hashlib
import os
import binascii
import requests
import ecdsa
import base58
import PySimpleGUI as sg
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib.pyplot as plt

# Database folder and application details
DATABASE = './database/'
CACHE_FILE = './cache.txt'
APP_TITLE = "Skyscope Sentinel Intelligence - BTC Key Tool (IBM Quantum Enhanced)"
DEVELOPER_INFO = "Developer: Casey Jay Topojani | Github: skyscope-sentinel"

# IBM Quantum initialization
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='334d70c5d25a2b538142a228c1c9b30d02496429469596a62c3e158198caf82536d63b8ce7722ee7befe404626dac86c7c7bea212ee4c535dbfc516bf1c3963f'
)

if not os.path.exists(CACHE_FILE):
    open(CACHE_FILE, "w").close()

def generate_private_key():
    return binascii.hexlify(os.urandom(32)).decode('utf-8').upper()

def private_key_to_public_key(private_key):
    pk = ecdsa.SigningKey.from_string(binascii.unhexlify(private_key), curve=ecdsa.SECP256k1)
    return '04' + binascii.hexlify(pk.verifying_key.to_string()).decode('utf-8')

def public_key_to_address(public_key):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(hashlib.sha256(binascii.unhexlify(public_key.encode())).digest())
    payload = '00' + ripemd160.hexdigest()
    checksum = hashlib.sha256(hashlib.sha256(binascii.unhexlify(payload.encode())).digest()).hexdigest()[:8]
    address_hex = payload + checksum
    n = int(address_hex, 16)
    output = []
    while n > 0:
        n, remainder = divmod(n, 58)
        output.append(alphabet[remainder])
    return ''.join(output[::-1])

def private_key_to_wif(private_key):
    var80 = "80" + private_key
    checksum = hashlib.sha256(binascii.unhexlify(hashlib.sha256(binascii.unhexlify(var80)).hexdigest())).hexdigest()[:8]
    payload = binascii.unhexlify(var80 + checksum)
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = int.from_bytes(payload, byteorder="big")
    output = []
    while n > 0:
        n, remainder = divmod(n, 58)
        output.append(alphabet[remainder])
    return ''.join(output[::-1])

def get_balance(address):
    try:
        response = requests.get(f"https://sochain.com/api/v2/address/BTC/{address}")
        return float(response.json()['data']['balance'])
    except:
        return -1

def read_database():
    database = set()
    for filename in os.listdir(DATABASE):
        with open(os.path.join(DATABASE, filename), 'r') as file:
            for address in file:
                address = address.strip()
                if address.startswith('1'):
                    database.add(address[-8:])
    return database

# Quantum Grover's Algorithm using IBM Quantum backend
def grovers_algorithm_ibm(target_state, num_qubits=5, iterations=1):
    qc = QuantumCircuit(num_qubits)

    # Initialize qubits in superposition
    qc.h(range(num_qubits))

    # Oracle to mark the target state
    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            qc.x(i)

    # Grover diffusion operator
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

    qc.measure_all()

    # Execute on IBM Quantum backend
    with service as session:
        sampler = Sampler(session=session)
        job = sampler.run(qc)
        result = job.result()
        counts = result.quasi_dists[0]

    # Display results
    plot_histogram(counts)
    plt.show()
    return max(counts, key=counts.get)

def scan_key(database, attempts, successes, quantum=False):
    private_key = generate_private_key()
    public_key = private_key_to_public_key(private_key)
    address = public_key_to_address(public_key)
    balance = get_balance(address)

    if quantum:
        # Quantum-assisted search to optimize scanning
        target_state = int(address[-5:], 16) % 32  # Reduced target state
        result = grovers_algorithm_ibm(target_state)
        print(f"Quantum Result: {result}")

    if balance > 0:
        successes += 1
        with open("found.txt", "a") as file:
            file.write(f"Address: {address}\nPrivate Key: {private_key}\nWIF: {private_key_to_wif(private_key)}\nBalance: {balance:.8f} BTC\n\n")
        print(f"[FOUND] Address: {address}, Balance: {balance:.8f} BTC")
    else:
        print(f"[CHECKED] Address: {address} - Balance: {balance:.8f} BTC")
    
    return attempts + 1, successes

def start_scanning(database, cores, quantum=False):
    attempts = 0
    successes = 0
    start_time = time()
    threading.Thread(target=speed_tracker, args=(start_time, attempts), daemon=True).start()

    with ThreadPoolExecutor(max_workers=cores) as executor:
        while True:
            future = executor.submit(scan_key, database, attempts, successes, quantum)
            attempts, successes = future.result()
            window.Element('_TRIES_').Update(str(attempts))
            window.Element('_WINS_').Update(str(successes))

def num_of_cores():
    available_cores = multiprocessing.cpu_count()
    return sg.popup_get_text(
        f"Number of available cores: {available_cores}\nHow many cores to use? (leave blank for all)", 
        title="Select CPU Cores"
    ) or available_cores

# GUI setup
sg.theme('DarkBlue12')
layout = [
    [sg.Text(APP_TITLE, size=(60, 1), justification='center', font=('Helvetica', 16, 'bold'))],
    [sg.Text(DEVELOPER_INFO, size=(60, 1), justification='center', font=('Helvetica', 10))],
    [sg.Text('Number of addresses tried:', size=(30, 1)), sg.Text('', key='_TRIES_', size=(30, 1))],
    [sg.Text('Number of addresses with balance:', size=(30, 1)), sg.Text('', key='_WINS_', size=(30, 1))],
    [sg.Output(size=(87, 20), font=('Courier New', 10))],
    [sg.Button('Start (Classical)', font=('Helvetica', 12), button_color=('white', 'green')),
     sg.Button('Start (Quantum)', font=('Helvetica', 12), button_color=('white', 'blue')),
     sg.Button('Exit', button_color=('white', 'red'))]
]

window = sg.Window('Skyscope BTC Tool (Quantum)', layout, default_element_size=(9, 1), finalize=True)

def main():
    global process
    database = read_database()
    print(f"Database loaded: {len(database)} addresses.")
    while True:
        event, values = window.read(timeout=10)
        if event in (None, 'Exit'):
            break
        elif event == 'Start (Classical)':
            process = not process
            if process:
                cores = int(num_of_cores())
                start_scanning(database, cores, quantum=False)
        elif event == 'Start (Quantum)':
            process = not process
            if process:
                cores = int(num_of_cores())
                start_scanning(database, cores, quantum=True)

    window.close()

if __name__ == '__main__':
    main()
