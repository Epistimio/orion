import React from 'react';
import Plot from 'react-plotly.js';

const config = {
  responsive: true,
};

export const LocalParameterImportancePlot = props => {
  return (
    <Plot
      id="lpi-plot"
      data={props.data}
      layout={props.layout}
      config={config}
      useResizeHandler={true}
      style={{ width: '100%' }}
    />
  );
};
