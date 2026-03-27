function SectorView({ data }) {
  const sectors = {};

  data.forEach(d => {
    if (!sectors[d.sector]) sectors[d.sector] = 0;
    sectors[d.sector]++;
  });

  return (
    <div>
      <h3>🏢 Sector Analysis</h3>
      {Object.keys(sectors).map(sec => (
        <div key={sec}>
          {sec}: {sectors[sec]}
        </div>
      ))}
    </div>
  );
}

export default SectorView;