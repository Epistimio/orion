import React from 'react';

export const PlotBox = props => {
  return (
    <article className="plot-box bx--col-md-4 bx--col-lg-4 bx--col-xlg-3 bx--offset-xlg-1">
      <h4 className="plot-box__heading">{props.heading}</h4>
      <div className="plot-box__body">{props.body}</div>
    </article>
  );
};
