import { useEffect, useState } from "react";
import AnomaliesTable from "./components/AnomaliesTable";
import AnomalyDetails from "./components/AnomalyDetails";

function App() {
  const [anomalies, setAnomalies] = useState([]);
  const [selectedAddress, setSelectedAddress] = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/anomalies", { mode: 'cors' })
      .then(res => res.json())
      .then(setAnomalies)
      .catch(console.error);
  }, []);

  return (
    <div style={{ padding: "20px", fontFamily: "monospace" }}>

      {
        selectedAddress ? (
          <>
            <button
              onClick={() => setSelectedAddress(false)}
            >
              Lista anomalie
            </button>

            <AnomalyDetails address={selectedAddress} />
          </>
        ) : (
          <>
            <h2>Anomalous Bitcoin Addresses</h2>

            <AnomaliesTable
              anomalies={anomalies}
              onSelect={setSelectedAddress}
            />
          </>
        )
      }
      
    </div>
  );
}

export default App;