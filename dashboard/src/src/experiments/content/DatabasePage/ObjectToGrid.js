import React from 'react';
import { Column, Grid, Row } from 'carbon-components-react';

/**
 * Component to pretty display an object (JSON dictionary) into data table.
 * Used to render trial parameters and statistics.
 */
export class ObjectToGrid extends React.Component {
  render() {
    const object = this.props.object;
    const keys = Object.keys(object);
    if (!keys.length) return '';
    keys.sort();
    return (
      <Grid condensed fullWidth className="object-to-grid">
        {keys.map(key => (
          <Row key={key}>
            <Column className="object-to-grid-key">
              <strong>
                <em>{key}</em>
              </strong>
            </Column>
            <Column>
              {Array.isArray(object[key])
                ? object[key].map((value, i) => (
                    <div key={i}>{value.toString()}</div>
                  ))
                : object[key].toString()}
            </Column>
          </Row>
        ))}
      </Grid>
    );
  }
}
