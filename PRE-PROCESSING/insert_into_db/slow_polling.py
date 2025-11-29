import time
import json
import os
import decimal
from bitcoinrpc.authproxy import AuthServiceProxy



SAVE_DIR = "./blocks"
os.makedirs(SAVE_DIR, exist_ok=True)

POLL_INTERVAL = 30  # secondi tra un controllo e l’altro


# --- encoder custom per Decimal ---
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)  # oppure: str(obj)
        return super(EnhancedJSONEncoder, self).default(obj)


def get_last_saved_height():
    files = [f for f in os.listdir(SAVE_DIR) if f.startswith("block_") and f.endswith(".json")]
    if not files:
        return -1
    return max(int(f.split("_")[1].split(".")[0]) for f in files)


def main():

    # Attendi RPC
    while True:
        try:
            rpc = AuthServiceProxy(f"http://{RPC_USER}:{RPC_PASS}@127.0.0.1:{RPC_PORT}")
            rpc.getblockchaininfo()
            break
        except:
            print("In attesa che Bitcoin Core sia disponibile...")
            time.sleep(5)

    print("RPC connesso!")

    while True:
        try:
            node_height = rpc.getblockcount()
            last_saved = get_last_saved_height()

            print(f"[INFO] Nodo: {node_height} | Salvato: {last_saved}")

            if last_saved >= node_height:
                time.sleep(POLL_INTERVAL)
                continue

            # Scarica nuovi blocchi
            for h in range(last_saved + 1, node_height + 1):
                block_hash = rpc.getblockhash(h)
                block = rpc.getblock(block_hash, 2)

                filepath = os.path.join(SAVE_DIR, f"block_{h}.json")

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(block, f, indent=2, cls=EnhancedJSONEncoder)

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("[ERRORE]", e)
            time.sleep(10)


if __name__ == "__main__":
    main()