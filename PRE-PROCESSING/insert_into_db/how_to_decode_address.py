import hashlib
from typing import Optional
import bech32  # pip install bech32

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b: bytes) -> str:
    """Codifica bytes in Base58Check."""
    num = int.from_bytes(b, byteorder='big')
    encode = ''
    while num > 0:
        num, rem = divmod(num, 58)
        encode = BASE58_ALPHABET[rem] + encode
    n_pad = 0
    for byte in b:
        if byte == 0:
            n_pad += 1
        else:
            break
    return '1' * n_pad + encode

def hash160(data: bytes) -> bytes:
    """SHA256 seguito da RIPEMD160."""
    return hashlib.new('ripemd160', hashlib.sha256(data).digest()).digest()

def pubkey_to_address(
    vout_hex: str,
    vout_type: str,
    vout_address: Optional[str] = None,
    mainnet: bool = True
) -> Optional[str]:
    """
    Converte un vout in un indirizzo Bitcoin.
    
    Args:
        vout_hex: esadecimale dello scriptPubKey
        vout_type: tipo di output (pubkey, pubkeyhash, scripthash, witness_v0_keyhash, witness_v0_scripthash, witness_v1_taproot, nulldata)
        vout_address: indirizzo già fornito (per pubkeyhash / scripthash)
        mainnet: True per mainnet, False per testnet
        
    Returns:
        Indirizzo Bitcoin o None se non spendibile / non gestito
    """
    if vout_type == "pubkey":
        # rimuove primo e ultimo byte (lunghezza chiave + OP_CHECKSIG)
        pubkey_hex = vout_hex[2:-2]
        pubkey_bytes = bytes.fromhex(pubkey_hex)
        h160 = hash160(pubkey_bytes)
        prefix = b'\x00' if mainnet else b'\x6f'
        payload = prefix + h160
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return base58_encode(payload + checksum)

    elif vout_type == "pubkeyhash":
        return vout_address  # già disponibile

    elif vout_type == "scripthash":
        # P2SH Base58Check
        if vout_address:
            return vout_address
        script_bytes = bytes.fromhex(vout_hex)
        h160 = hashlib.new('ripemd160', hashlib.sha256(script_bytes).digest()).digest()
        prefix = b'\x05' if mainnet else b'\xc4'
        payload = prefix + h160
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        return base58_encode(payload + checksum)

    elif vout_type in ("witness_v0_keyhash", "witness_v0_scripthash"):
        # Bech32 indirizzi SegWit
        hrp = "bc" if mainnet else "tb"
        # vout_hex contiene solo l'hash (ripulire se necessario)
        script_bytes = bytes.fromhex(vout_hex)
        version = 0
        # convert bytes to 5-bit array
        data = bech32.convertbits(script_bytes, 8, 5)
        return bech32.bech32_encode(hrp, [version] + data)

    elif vout_type == "witness_v1_taproot":
        # Bech32m indirizzi Taproot
        hrp = "bc" if mainnet else "tb"
        script_bytes = bytes.fromhex(vout_hex)
        version = 1
        data = bech32.convertbits(script_bytes, 8, 5)
        return bech32.bech32m_encode(hrp, [version] + data)

    elif vout_type == "nulldata":
        return None  # OP_RETURN, non spendibile

    else:
        print(f"[WARN] Tipo vout non gestito: {vout_type}")
        return None