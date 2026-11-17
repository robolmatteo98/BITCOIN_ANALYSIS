import hashlib
import struct
from pathlib import Path

def parse_binary_file_record(item):
    path, payload = item

    if hasattr(payload, "read"):
        payload = payload.read()

    return parse_dat_payload(path, payload)

MAINNET_MAGIC = bytes.fromhex("f9beb4d9")

def double_sha256(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def hash160(data):
    sha = hashlib.sha256(data).digest()
    ripe = hashlib.new("ripemd160")
    ripe.update(sha)
    return ripe.digest()

def hash256_hex(data):
    return double_sha256(data)[::-1].hex()

def read_varint(buffer, offset):
    first = buffer[offset]
    if first < 0xFD:
        return first, offset + 1
    if first == 0xFD:
        return struct.unpack_from("<H", buffer, offset + 1)[0], offset + 3
    if first == 0xFE:
        return struct.unpack_from("<I", buffer, offset + 1)[0], offset + 5
    return struct.unpack_from("<Q", buffer, offset + 1)[0], offset + 9

def base58check(prefix, payload):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    data = prefix + payload
    checksum = double_sha256(data)[:4]
    value = int.from_bytes(data + checksum, "big")
    encoded = ""
    while value > 0:
        value, mod = divmod(value, 58)
        encoded = alphabet[mod] + encoded
    leading_zeroes = len(data + checksum) - len((data + checksum).lstrip(b"\x00"))
    return "1" * leading_zeroes + encoded

def bech32_polymod(values):
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            if (top >> i) & 1:
                chk ^= generator[i]
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_create_checksum(hrp, data, spec):
    const = 0x2BC830A3 if spec == "bech32m" else 1
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ const
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def bech32_encode(hrp, data, spec):
    charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    combined = data + bech32_create_checksum(hrp, data, spec)
    return hrp + "1" + "".join(charset[d] for d in combined)

def convertbits(data, from_bits, to_bits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << to_bits) - 1
    max_acc = (1 << (from_bits + to_bits - 1)) - 1
    for value in data:
        if value < 0 or value >> from_bits:
            return None
        acc = ((acc << from_bits) | value) & max_acc
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        return None
    return ret

def segwit_address(version, program):
    data = [version] + convertbits(program, 8, 5)
    spec = "bech32m" if version > 0 else "bech32"
    return bech32_encode("bc", data, spec)

def parse_script_pubkey(script):
    result = {
        "address": None,
        "entity": None,
    }

    if len(script) == 25 and script[:3] == b"\x76\xa9\x14" and script[-2:] == b"\x88\xac":
        payload = script[3:23]
        address = base58check(b"\x00", payload)
        result.update({"address": address, "entity": address})
        return result

    if len(script) == 23 and script[:2] == b"\xa9\x14" and script[-1:] == b"\x87":
        payload = script[2:22]
        address = base58check(b"\x05", payload)
        result.update({"address": address, "entity": address})
        return result

    if len(script) == 22 and script[0] == 0x00 and script[1] == 0x14:
        address = segwit_address(0, script[2:])
        result.update({"address": address, "entity": address})
        return result

    if len(script) == 34 and script[0] == 0x00 and script[1] == 0x20:
        address = segwit_address(0, script[2:])
        result.update({"address": address, "entity": address})
        return result

    if len(script) == 34 and script[0] == 0x51 and script[1] == 0x20:
        address = segwit_address(1, script[2:])
        result.update({"address": address, "entity": address})
        return result

    if len(script) in (35, 67) and script[-1] == 0xAC:
        pubkey_bytes = script[1:-1]
        address = base58check(b"\x00", hash160(pubkey_bytes))
        result.update({
            "address": address,
            "entity": address,
        })
        return result

    return result

def parse_transaction(buffer, offset, prevout_index):
    version_bytes = buffer[offset:offset + 4]
    offset += 4

    marker = buffer[offset] if offset < len(buffer) else None
    flag = buffer[offset + 1] if offset + 1 < len(buffer) else None
    is_segwit = marker == 0 and flag == 1
    if is_segwit:
        offset += 2

    stripped_parts = [version_bytes]

    vin_count, offset = read_varint(buffer, offset)
    stripped_parts.append(_encode_varint(vin_count))
    vin = []
    for _ in range(vin_count):
        prev_tx_raw = buffer[offset:offset + 32]
        prev_txid = prev_tx_raw[::-1].hex()
        offset += 32

        prev_vout_raw = buffer[offset:offset + 4]
        prev_vout = struct.unpack("<I", prev_vout_raw)[0]
        offset += 4

        script_len, offset = read_varint(buffer, offset)
        script_sig = buffer[offset:offset + script_len]
        offset += script_len

        sequence_raw = buffer[offset:offset + 4]
        sequence = struct.unpack("<I", sequence_raw)[0]
        offset += 4

        stripped_parts.extend(
            [prev_tx_raw, prev_vout_raw, _encode_varint(script_len), script_sig, sequence_raw]
        )

        if prev_txid == "0" * 64 and prev_vout == 0xFFFFFFFF:
            vin.append({})
        else:
            prevout = prevout_index.get((prev_txid, prev_vout))
            vin.append({
                "txid": prev_txid,
                "vout": prev_vout,
                "prevout_entity": prevout["entity"] if prevout else None,
                "prevout_address": prevout["address"] if prevout else None,
            })

    vout_count, offset = read_varint(buffer, offset)
    stripped_parts.append(_encode_varint(vout_count))
    vout = []
    for index in range(vout_count):
        value_raw = buffer[offset:offset + 8]
        value_sat = struct.unpack("<q", value_raw)[0]
        offset += 8

        script_len, offset = read_varint(buffer, offset)
        script_pubkey = buffer[offset:offset + script_len]
        offset += script_len

        stripped_parts.extend([value_raw, _encode_varint(script_len), script_pubkey])

        parsed_script = parse_script_pubkey(script_pubkey)
        output = {
            "n": index,
            "value_sat": value_sat,
            "value_btc": value_sat / 100_000_000,
            "scriptPubKey": parsed_script,
        }
        vout.append(output)

    if is_segwit:
        for _ in range(vin_count):
            witness_items, offset = read_varint(buffer, offset)
            for _ in range(witness_items):
                item_len, offset = read_varint(buffer, offset)
                offset += item_len

    locktime_raw = buffer[offset:offset + 4]
    locktime = struct.unpack("<I", locktime_raw)[0]
    offset += 4
    stripped_parts.append(locktime_raw)

    txid = hash256_hex(b"".join(stripped_parts))
    for output in vout:
        prevout_index[(txid, output["n"])] = output["scriptPubKey"]

    return {
        "txid": txid,
        "vin": vin,
        "vout": vout,
    }, offset

def _encode_varint(value):
    if value < 0xFD:
        return bytes([value])
    if value <= 0xFFFF:
        return b"\xFD" + struct.pack("<H", value)
    if value <= 0xFFFFFFFF:
        return b"\xFE" + struct.pack("<I", value)
    return b"\xFF" + struct.pack("<Q", value)

def parse_block(block_bytes, dat_file, offset_in_file, raw_size, prevout_index):
    prev_block = block_bytes[4:36][::-1].hex()
    timestamp = struct.unpack_from("<I", block_bytes, 68)[0]
    block_hash = hash256_hex(block_bytes[:80])

    offset = 80
    tx_count, offset = read_varint(block_bytes, offset)
    tx = []
    for _ in range(tx_count):
        parsed_tx, offset = parse_transaction(block_bytes, offset, prevout_index)
        tx.append(parsed_tx)

    return {
        "source_file": dat_file,
        "file_offset": offset_in_file,
        "hash": block_hash,
        "previousblockhash": prev_block,
        "time": timestamp,
        "tx": tx,
    }

def iter_blocks_from_bytes(dat_file_name, payload):
    prevout_index = {}
    offset = 0
    size = len(payload)

    while offset + 8 <= size:
        start_offset = offset
        magic = payload[offset:offset + 4]
        offset += 4
        if magic != MAINNET_MAGIC:
            raise ValueError(f"Magic bytes non validi in {dat_file_name} a offset {start_offset}: {magic.hex()}")

        block_size = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        block_bytes = payload[offset:offset + block_size]
        offset += block_size
        if len(block_bytes) != block_size:
            break

        yield parse_block(block_bytes, dat_file_name, start_offset, block_size, prevout_index)

def infer_heights(blocks):
    by_hash = {block["hash"]: block for block in blocks}
    zero_hash = "0" * 64
    for block in blocks:
        block["height"] = None

    changed = True
    while changed:
        changed = False
        for block in blocks:
            if block["height"] is not None:
                continue
            if block["previousblockhash"] == zero_hash:
                block["height"] = 0
                changed = True
                continue
            parent = by_hash.get(block["previousblockhash"])
            if parent and parent.get("height") is not None:
                block["height"] = parent["height"] + 1
                changed = True

def parse_dat_payload(dat_path, payload):
    dat_file_name = Path(dat_path).name
    blocks = list(iter_blocks_from_bytes(dat_file_name, payload))
    infer_heights(blocks)
    return blocks

def parse_local_dat_file(dat_path):
    path = Path(dat_path)
    return parse_dat_payload(path, path.read_bytes())
