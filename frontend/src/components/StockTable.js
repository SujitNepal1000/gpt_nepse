function StockTable({ data }) {
  return (
    <table border="1" width="100%">
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Sector</th>
          <th>LTP</th>
          <th>RSI</th>
          <th>Signal</th>
          <th>Entry</th>
          <th>Target</th>
          <th>SL</th>
        </tr>
      </thead>
      <tbody>
        {data.map((d, i) => (
          <tr key={i}>
            <td>{d.symbol}</td>
            <td>{d.sector}</td>
            <td>{d.ltp}</td>
            <td>{d.rsi}</td>
            <td>{d.signal}</td>
            <td>{d.entry}</td>
            <td>{d.target}</td>
            <td>{d.stoploss}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default StockTable;