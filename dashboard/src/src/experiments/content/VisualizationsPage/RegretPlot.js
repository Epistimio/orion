import React from 'react';
import Plot from 'react-plotly.js';

const config = {
  responsive: true,
};

export function RegretConst(props) {
  return (
    <Plot
      id="regret-plot"
      data={props.data}
      layout={props.layout}
      config={config}
      useResizeHandler={true}
      style={{ width: '100%' }}
    />
  );
}
