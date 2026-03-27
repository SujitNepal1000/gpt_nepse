import React, { useEffect, useState } from "react";
import { getStocks } from "../services/api";
import StockTable from "./StockTable";
import Filters from "./Filters";
import Signals from "./Signals";
import SectorView from "./SectorView";

function Dashboard() {
  const [data, setData] = useState([]);
  const [filtered, setFiltered] = useState([]);

  useEffect(() => {
    getStocks().then(res => {
      setData(res.data);
      setFiltered(res.data);
    });
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>📊 NEPSE Dashboard</h2>

      <Filters data={data} setFiltered={setFiltered} />

      <Signals data={filtered} />

      <SectorView data={filtered} />

      <StockTable data={filtered} />
    </div>
  );
}

export default Dashboard;