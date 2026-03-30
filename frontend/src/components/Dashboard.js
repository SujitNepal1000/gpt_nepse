import React, { useEffect, useState } from "react";
import { getStocks } from "../services/api";
import StockTable from "./StockTable";
import Filters from "./Filters";
import Signals from "./Signals";
import SectorView from "./SectorView";
import UploadExcel from "./UploadExcel";

function Dashboard() {
  const [data, setData] = useState([]);
  const [filtered, setFiltered] = useState([]);

  const loadData = () => {
    getStocks().then(res => {
      setData(res.data);
      setFiltered(res.data);
    });
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>📊 NEPSE Dashboard</h2>

      <UploadExcel onUploadSuccess={loadData} />

      <Filters data={data} setFiltered={setFiltered} />

      <Signals data={filtered} />

      <SectorView data={filtered} />

      <StockTable data={filtered} />
    </div>
  );
}

export default Dashboard;