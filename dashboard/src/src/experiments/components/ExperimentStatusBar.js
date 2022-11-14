import React from 'react';
import PropTypes from 'prop-types';
import { Backend, DEFAULT_BACKEND } from '../../utils/queryServer';
import ProgressBar from 'react-bootstrap/ProgressBar';
import {
  Column,
  Grid,
  Row,
} from 'carbon-components-react';

function floatToPercent(value) {
  return Math.round(value * 100);
}

export class ExperimentStatusBar extends React.Component {
  _isMounted = false;
  constructor(props) {
    super(props);
    this.state = { status: null };
  }
  render() {
    if (this.state.status === null)
      return `Loading status bar for experiment "${this.props.name}" ...`;
    if (this.state.status === false)
      return `Unable to load status bar for experiment "${this.props.name}".`;
    const pc = this.state.status.trial_status_percentage;
    return (
      <div>
        <Grid className="pb-2">
          <Row>
            <Column>
              <strong>Execution time</strong>:&nbsp;
              <code>{this.state.status.current_execution_time}</code>
            </Column>
            <Column className="justify-content-center">
              <strong>Whole clock time</strong>:&nbsp;
              <code>{this.state.status.whole_clock_time} %</code>
            </Column>
            <Column className="justify-content-end">
              <strong>ETA</strong>:&nbsp;<code>{this.state.status.eta}</code>
            </Column>
          </Row>
        </Grid>
        <div className="pb-2">
          <ProgressBar>
            <ProgressBar
              variant="success"
              now={floatToPercent(pc.completed)}
              key={1}
            />
            <ProgressBar
              variant="warning"
              now={floatToPercent(pc.interrupted)}
              key={2}
            />
            <ProgressBar
              variant="danger"
              now={floatToPercent(pc.broken)}
              key={3}
            />
            <ProgressBar
              variant="info"
              now={floatToPercent(pc.suspended)}
              key={4}
            />
          </ProgressBar>
        </div>
        <Grid className="pb-2">
          <Row>
            <Column className="justify-content-end">
              <strong>Progress</strong>:&nbsp;
              <code>{floatToPercent(this.state.status.progress)} %</code>
            </Column>
          </Row>
        </Grid>
      </div>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    const backend = new Backend(DEFAULT_BACKEND);
    backend
      .query(`experiments/status/${this.props.name}`)
      .then(status => {
        console.log(status);
        if (this._isMounted) {
          this.setState({ status });
        }
      })
      .catch(error => {
        console.error(error);
        if (this._isMounted) {
          this.setState({ status: false });
        }
      });
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
}
ExperimentStatusBar.propTypes = {
  name: PropTypes.string.isRequired,
};
