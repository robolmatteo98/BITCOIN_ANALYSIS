import hashlib

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b: bytes) -> str:
    """Codifica bytes in Base58Check."""
    num = int.from_bytes(b, byteorder='big')
    encode = ''
    while num > 0:
        num, rem = divmod(num, 58)
        encode = BASE58_ALPHABET[rem] + encode
    # aggiunge '1' per ogni byte iniziale 0
    n_pad = 0
    for byte in b:
        if byte == 0:
            n_pad += 1
        else:
            break
    return '1' * n_pad + encode

def pubkey_to_address(vout_hex: str, mainnet: bool = True) -> str:
    """
    Estrae la chiave pubblica dal vout.hex e calcola l'indirizzo Bitcoin.
    
    Args:
        vout_hex: il campo "hex" della transazione vout (coinbase/pubkey)
        mainnet: True per mainnet, False per testnet

    Returns:
        Indirizzo Bitcoin in formato Base58Check
    """
    # rimuove il primo byte (lunghezza) e l'ultimo byte (OP_CHECKSIG)
    pubkey_hex = vout_hex[2:-2]  
    pubkey_bytes = bytes.fromhex(pubkey_hex)

    # hash SHA256
    sha256 = hashlib.sha256(pubkey_bytes).digest()
    # hash RIPEMD-160
    ripemd160 = hashlib.new('ripemd160', sha256).digest()
    # prefisso mainnet/testnet
    prefix = b'\x00' if mainnet else b'\x6f'
    payload = prefix + ripemd160
    # checksum
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    address_bytes = payload + checksum
    # Base58Check finale
    return base58_encode(address_bytes)