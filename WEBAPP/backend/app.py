from flask import Flask, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

with open("./data/anomalies.json") as f:
    data = json.load(f)

anomalies = data["anomalies"]
anomalies_by_address = {a["address"]: a for a in anomalies}

@app.route("/api/anomalies")
def get_anomalies():
    return jsonify([
        {
            "address": a["address"],
            "anomaly_score": a["anomaly_score"],
            "reasons": [r["label"] for r in a["reasons"]],
            "stats": a["stats"],
            "num_edges": len(a["edges"])
        }
        for a in anomalies
    ])

@app.route("/api/anomalies/<address>")
def get_anomaly_details(address):
    anomaly = anomalies_by_address.get(address)
    if not anomaly:
        return jsonify({"error": "Not found"}), 404
    return jsonify(anomaly)

if __name__ == "__main__":
    app.run(debug=True)