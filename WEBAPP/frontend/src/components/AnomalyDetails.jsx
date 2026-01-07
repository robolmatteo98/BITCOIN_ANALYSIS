import { useEffect, useState } from "react";

export default function AnomalyDetails({ address }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch(`http://127.0.0.1:5000/api/anomalies/${address}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [address]);

  if (!data) return <p>Loading details...</p>;

  // Genera una mappa txid -> colore alternato
  const colors = ["#f0f8ff33", "#e6ffe60f"]; // puoi cambiare i colori
  const txidColorMap = {};
  let colorIndex = 0;

  data.edges.forEach(edge => {
    if (!(edge.txid in txidColorMap)) {
      txidColorMap[edge.txid] = colors[colorIndex % colors.length];
      colorIndex++;
    }
  });

  return (
    <div>
      <h3>Details for {address}</h3>
      <p>
        <strong>Anomaly score:</strong> {data.anomaly_score.toFixed(3)}
      </p>
      <p>
        <strong>Reasons:</strong>{" "}
        {data.reasons.map(r => r.label).join(", ")}
      </p>

      <h4>Transactions</h4>
      <table border="1" cellPadding="6" style={{ width: "100%" }}>
        <thead>
          <tr>
            <th>TxID</th>
            <th>From</th>
            <th>To</th>
            <th>Amount</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {data.edges.map((e, i) => (
            <tr
              key={i}
              style={{ backgroundColor: txidColorMap[e.txid] }}
            >
              <td>{e.txid}</td>
              <td>{e.from_address}</td>
              <td>{e.to_address}</td>
              <td>{e.flow_amount}</td>
              <td>{new Date(e.time * 1000).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}