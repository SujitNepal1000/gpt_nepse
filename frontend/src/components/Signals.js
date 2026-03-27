function Signals({ data }) {
  const strong = data.filter(d => d.signal === "STRONG BUY");

  return (
    <div>
      <h3>🚀 High Potential Stocks</h3>
      {strong.map((s, i) => (
        <div key={i}>
          {s.symbol} → Entry: {s.entry} | Target: {s.target}
        </div>
      ))}
    </div>
  );
}

export default Signals;