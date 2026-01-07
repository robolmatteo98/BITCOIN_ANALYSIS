import { useState, useMemo } from "react";

export default function AnomaliesTable({ anomalies, onSelect }) {
  const [sortAsc, setSortAsc] = useState(false); // true = crescente
  const [reasonFilter, setReasonFilter] = useState(""); // filtro reasons

  // Raccogli tutti i valori unici di reasons per il select
  const allReasons = useMemo(() => {
    const set = new Set();
    anomalies.forEach(a => a.reasons.forEach(r => set.add(r)));
    return Array.from(set);
  }, [anomalies]);

  // Applica filtro e ordinamento
  const filteredAnomalies = useMemo(() => {
    let result = [...anomalies];
    if (reasonFilter) {
      result = result.filter(a => a.reasons.includes(reasonFilter));
    }
    result.sort((a, b) =>
      sortAsc
        ? a.anomaly_score - b.anomaly_score
        : b.anomaly_score - a.anomaly_score
    );
    return result;
  }, [anomalies, reasonFilter, sortAsc]);

  return (
    <div style={{ marginBottom: "20px" }}>
      {/* Filtro Reasons */}
      <label>
        Filter by Reason:{" "}
        <select
          value={reasonFilter}
          onChange={e => setReasonFilter(e.target.value)}
        >
          <option value="">All</option>
          {allReasons.map(r => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </label>

      <table border="1" cellPadding="6" style={{ width: "100%", marginTop: "10px" }}>
        <thead>
          <tr>
            <th>Address</th>
            <th
              style={{ cursor: "pointer" }}
              onClick={() => setSortAsc(!sortAsc)}
            >
              Score {sortAsc ? "▲" : "▼"}
            </th>
            <th>Reasons</th>
            <th>Out Deg</th>
            <th>In Deg</th>
            <th>#Edges</th>
          </tr>
        </thead>
        <tbody>
          {filteredAnomalies.map(a => (
            <tr
              key={a.address}
              style={{ cursor: "pointer" }}
              onClick={() => onSelect(a.address)}
            >
              <td>{a.address}</td>
              <td>{a.anomaly_score.toFixed(3)}</td>
              <td>{a.reasons.join(", ")}</td>
              <td>{a.stats.out_degree}</td>
              <td>{a.stats.in_degree}</td>
              <td>{a.num_edges}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}