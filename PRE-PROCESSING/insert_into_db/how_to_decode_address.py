import hashlib

# Funzione per Base58Check manuale (senza usare librerie esterne)
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b):
    # Converti bytes in intero
    num = int.from_bytes(b, byteorder='big')
    # Codifica in Base58
    encode = ''
    while num > 0:
        num, rem = divmod(num, 58)
        encode = BASE58_ALPHABET[rem] + encode
    # Aggiungi '1' per ogni byte iniziale 0
    n_pad = 0
    for byte in b:
        if byte == 0:
            n_pad += 1
        else:
            break
    return '1' * n_pad + encode

def pubkey_to_address(pubkey_hex, mainnet=True):
    pubkey_bytes = bytes.fromhex(pubkey_hex) # trasformo in bytes
    sha256 = hashlib.sha256(pubkey_bytes).digest() # calcolo l'hash in 256 bit
    ripemd160 = hashlib.new('ripemd160', sha256).digest() # produce un altro hash di 160 bit
    prefix = b'\x00' if mainnet else b'\x6f' # aggiungo un prefisso di rete
    payload = prefix + ripemd160
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] # firma di controllo dell'indirizzo (se ci sono stati errori)
    address_bytes = payload + checksum
    return base58_encode(address_bytes) # codifico in base 58