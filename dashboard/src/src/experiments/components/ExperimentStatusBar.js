import React from 'react';
import PropTypes from 'prop-types';
import { Backend, DEFAULT_BACKEND } from '../../utils/queryServer';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { Column, Grid, Row, Tooltip } from 'carbon-components-react';

function floatToPercent(value) {
  return Math.round(value * 100);
}

const StatusToProgress = {
  new: 'new', // NB: user-made / we don't need to display trial status "new" in progress bar
  reserved: 'info',
  suspended: 'suspended', // user-made
  completed: 'success',
  interrupted: 'warning',
  broken: 'danger',
};

export class ExperimentStatusBar extends React.Component {
  _isMounted = false;
  constructor(props) {
    super(props);
    this.state = { status: null };
  }
  render() {
    if (this.state.status === null)
      return (
        <div>
          <ProgressBar
            now={100}
            variant="running"
            animated
            title={`Loading status bar for experiment "${this.props.name}" ...`}
            key={1}
          />
        </div>
      );
    if (this.state.status === false)
      return (
        <div>
          <ProgressBar
            variant="danger"
            label="error"
            now={100}
            title={`Unable to load status bar for experiment "${this.props.name}".`}
          />
        </div>
      );
    const sc = this.state.status.trial_status_count;
    const nb = this.state.status.nb_trials;
    return (
      <div>
        {this.props.withInfo ? (
          <Grid className="pb-2">
            <Row>
              <Column>
                <strong>Execution time</strong>:&nbsp;
                <code>{this.state.status.current_execution_time}</code>
                <Tooltip>Current execution time for all completed trials</Tooltip>
              </Column>
              <Column className="justify-content-center">
                <strong>Whole clock time</strong>:&nbsp;
                <code>{this.state.status.whole_clock_time}</code>
                <Tooltip>Sum of trials execution time</Tooltip>
              </Column>
              <Column className="justify-content-end">
                <strong>ETA</strong>:&nbsp;<code>{this.state.status.eta}</code>
                <Tooltip>Estimated time for experiment to finish</Tooltip>
              </Column>
            </Row>
          </Grid>
        ) : (
          ''
        )}
        <div {...(this.props.withInfo ? { className: 'pb-2' } : {})}>
          <ProgressBar>
            <ProgressBar
              variant={StatusToProgress.completed}
              now={floatToPercent(sc.completed / nb)}
              key={1}
            />
            <ProgressBar
              variant={StatusToProgress.suspended}
              now={floatToPercent(sc.suspended / nb)}
              key={2}
            />
            <ProgressBar
              variant={StatusToProgress.interrupted}
              now={floatToPercent(sc.interrupted / nb)}
              key={2}
            />
            <ProgressBar
              variant={StatusToProgress.broken}
              now={floatToPercent(sc.broken / nb)}
              key={3}
            />
            <ProgressBar
              variant={StatusToProgress.reserved}
              now={floatToPercent(sc.reserved / nb)}
              key={4}
            />
          </ProgressBar>
        </div>
        {this.props.withInfo ? (
          <Grid className="pb-2">
            <Row>
              <Column className="justify-content-end">
                <strong>Progress</strong>:&nbsp;
                <code>{floatToPercent(this.state.status.progress)} %</code>
                <Tooltip>Experiment progression percentage</Tooltip>
              </Column>
            </Row>
          </Grid>
        ) : (
          ''
        )}
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
  withInfo: PropTypes.bool,
};
