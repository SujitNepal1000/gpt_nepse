import React, { useState } from "react";

function Filters({ data, setFiltered }) {
  const [signal, setSignal] = useState("");

  const applyFilter = () => {
    let filtered = data;

    if (signal) {
      filtered = filtered.filter(d => d.signal === signal);
    }

    setFiltered(filtered);
  };

  return (
    <div>
      <select onChange={(e) => setSignal(e.target.value)}>
        <option value="">All</option>
        <option value="STRONG BUY">Strong Buy</option>
        <option value="BUY">Buy</option>
      </select>

      <button onClick={applyFilter}>Apply</button>
    </div>
  );
}

export default Filters;