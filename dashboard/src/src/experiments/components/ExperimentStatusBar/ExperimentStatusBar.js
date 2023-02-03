import React from 'react';
import PropTypes from 'prop-types';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { Column, Grid, Row, Tooltip } from 'carbon-components-react';
import { EXPERIMENT_STATUS_PROVIDER } from './ExperimentStatusProvider';

function floatToPercent(value) {
  return Math.round(value * 100);
}

export const StatusToProgress = {
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
    this.onFocus = this.onFocus.bind(this);
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
    return (
      <div>
        {this.props.withInfo ? this.renderExperimentInfo() : ''}
        {this.props.withInfo ? (
          <Grid fullWidth className="mb-2">
            <Row>
              <Column>
                <strong>Elapsed time</strong>:&nbsp;
                <code>{this.state.status.duration}</code>
                <Tooltip>
                  Time elapsed since the beginning of the HPO execution
                </Tooltip>
              </Column>
              <Column className="text-sm-center">
                <strong>Sum of trials time</strong>:&nbsp;
                <code>{this.state.status.whole_clock_time}</code>
                <Tooltip>Sum of trials execution time</Tooltip>
              </Column>
              <Column className="text-sm-right">
                <strong>ETA</strong>:&nbsp;
                <code>
                  {this.state.status.eta === null
                    ? '(unknown)'
                    : this.state.status.eta === 0
                    ? 0
                    : this.state.status.eta === 'infinite'
                    ? '\u221E'
                    : `${this.state.status.eta} @ ${new Date(
                        new Date().getTime() +
                          this.state.status.eta_milliseconds
                      ).toLocaleString()}`}
                </code>
                <Tooltip>Estimated time for experiment to finish</Tooltip>
              </Column>
            </Row>
          </Grid>
        ) : (
          ''
        )}
        <div {...(this.props.withInfo ? { className: 'mb-2' } : {})}>
          {this.state.status.max_trials === 'infinite' ? (
            <ProgressBar
              striped={true}
              label={'N/A'}
              title={'N/A (max trials \u221E)'}
              now={100}
              variant="running"
            />
          ) : (
            <ProgressBar>
              {[
                'completed',
                'suspended',
                'interrupted',
                'broken',
                'reserved',
              ].map((trialStatus, i) =>
                this.renderProgressPart(trialStatus, i)
              )}
            </ProgressBar>
          )}
        </div>
        {this.props.withInfo ? (
          <Grid className="mb-4">
            <Row>
              <Column className="d-flex flex-row">
                {[
                  'completed',
                  'suspended',
                  'interrupted',
                  'broken',
                  'reserved',
                ].map((trialStatus, i) =>
                  this.renderLegendPart(trialStatus, i)
                )}
              </Column>
              <Column className="text-sm-right">
                <strong>Progress</strong>:&nbsp;
                <code>
                  {this.state.status.progress === null
                    ? '(unknown)'
                    : `${floatToPercent(this.state.status.progress)} %`}
                </code>
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
  renderExperimentInfo() {
    const stats = this.state.status;
    const rows = [
      {
        title: 'Best trial ID',
        value:
          stats.best_trials_id === null ? '(unknown)' : stats.best_trials_id,
      },
      {
        title: 'Best evaluation',
        value:
          stats.best_evaluation === null ? '(unknown)' : stats.best_evaluation,
      },
      { title: 'Start time', value: stats.start_time },
      { title: 'Finish time', value: stats.finish_time },
      { title: 'Trials', value: stats.nb_trials },
      {
        title: 'Max trials',
        value: stats.max_trials === 'infinite' ? '\u221E' : stats.max_trials,
      },
    ];
    return (
      <div>
        <h3 className="text-center mb-2">Experiment "{this.props.name}"</h3>
        <Grid className="mb-3">
          {rows.map((row, indexRow) => (
            <Row key={indexRow}>
              <Column className="text-right">
                <strong>{row.title}</strong>
              </Column>
              <Column>{row.value}</Column>
            </Row>
          ))}
        </Grid>
      </div>
    );
  }
  renderProgressPart(trialStatus, key) {
    const stats = this.state.status;
    const progressBase = Math.max(stats.max_trials, stats.nb_trials);
    return (
      <ProgressBar
        key={key}
        variant={StatusToProgress[trialStatus]}
        now={floatToPercent(
          (this.state.status.trial_status_count[trialStatus] || 0) /
            progressBase
        )}
        title={`${trialStatus} (${this.state.status.trial_status_count[
          trialStatus
        ] || 0})`}
        onClick={() => this.onFocus(trialStatus)}
        striped={this.props.focus === trialStatus}
      />
    );
  }
  renderLegendPart(trialStatus, key) {
    return (
      <div key={key} className="d-flex flex-row align-items-baseline">
        <ProgressBar
          now={100}
          style={{ width: '1rem' }}
          variant={StatusToProgress[trialStatus]}
        />
        <div className="px-1">
          {trialStatus.charAt(0).toUpperCase()}
          {trialStatus.slice(1)} (
          {this.state.status.trial_status_count[trialStatus] || 0})
        </div>
      </div>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    EXPERIMENT_STATUS_PROVIDER.get(this.props.name)
      .then(status => {
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
  onFocus(trialStatus) {
    if (this.props.onFocus) {
      this.props.onFocus(trialStatus);
    }
  }
}
ExperimentStatusBar.propTypes = {
  name: PropTypes.string.isRequired,
  withInfo: PropTypes.bool,
  focus: PropTypes.string,
  onFocus: PropTypes.func,
};
