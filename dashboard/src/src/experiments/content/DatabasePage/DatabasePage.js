import React from 'react';
import { Backend, DEFAULT_BACKEND } from '../../../utils/queryServer';
import TrialTable from './TrialTable';
import { BackendContext } from '../../BackendContext';
import { Grid, Row, Column } from 'carbon-components-react';

/**
 * Component to pretty display an object (JSON dictionary) into data table.
 * Used to render trial parameters and statistics.
 */
class ObjectToGrid extends React.Component {
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

/**
 * Utility class to provide and cache experiment trials.
 */
class TrialsProvider {
  constructor(address) {
    this.backend = new Backend(address);
    this.trials = {};
  }
  async get(experiment) {
    if (!this.trials.hasOwnProperty(experiment)) {
      const queryTrials = await this.backend.query(`trials/${experiment}`);
      const trialIndices = queryTrials.map(trial => trial.id);
      trialIndices.sort();
      const trials = [];
      for (let trialID of trialIndices) {
        const trial = await this.backend.query(
          `trials/${experiment}/${trialID}`
        );
        // Save parameters and statistics as already rendered components.
        trial.parameters = <ObjectToGrid object={trial.parameters} />;
        trial.statistics = <ObjectToGrid object={trial.statistics} />;
        trials.push(trial);
      }
      this.trials[experiment] = trials;
    }
    return this.trials[experiment];
  }
}

/**
 * Singleton to provide experiment trials.
 * @type {TrialsProvider}
 */
const TRIALS_PROVIDER = new TrialsProvider(DEFAULT_BACKEND);

const headers = [
  {
    key: 'id',
    header: 'ID',
  },
  {
    key: 'submitTime',
    header: 'Submit time',
  },
  {
    key: 'startTime',
    header: 'Start time',
  },
  {
    key: 'endTime',
    header: 'End time',
  },
  {
    key: 'parameters',
    header: 'Parameters',
  },
  {
    key: 'objective',
    header: 'Objective',
  },
  {
    key: 'statistics',
    header: 'Statistics',
  },
];

class DatabasePage extends React.Component {
  // Control variable to avoid setting state if component was unmounted before an asynchronous API call finished.
  _isMounted = false;
  static contextType = BackendContext;
  constructor(props) {
    super(props);
    this.state = { experiment: null, trials: null };
  }
  render() {
    if (this.state.experiment === null)
      return 'No trials to display, please select an experiment.';
    if (this.state.trials === null)
      return `Loading trials for experiment "${this.state.experiment}" ...`;
    if (this.state.trials === false)
      return `Unable to load trials for experiment "${this.state.experiment}".`;
    return (
      <div className="bx--grid bx--grid--full-width bx--grid--no-gutter database-page">
        <div className="bx--row database-page__r1">
          <div className="bx--col-lg-16">
            <TrialTable
              headers={headers}
              rows={this.state.trials}
              experiment={this.state.experiment}
            />
          </div>
        </div>
      </div>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    const experiment = this.context.experiment;
    if (experiment !== null) {
      this.loadTrials(experiment);
    }
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    // We must check if selected experiment changed
    const experiment = this.context.experiment;
    if (this.state.experiment !== experiment) {
      if (experiment === null) {
        this.setState({ experiment, trials: null });
      } else {
        this.loadTrials(experiment);
      }
    }
  }
  loadTrials(experiment) {
    this.setState({ experiment, trials: null }, () => {
      TRIALS_PROVIDER.get(experiment)
        .then(trials => {
          if (this._isMounted) {
            this.setState({ trials });
          }
        })
        .catch(error => {
          console.error(error);
          if (this._isMounted) {
            this.setState({ trials: false });
          }
        });
    });
  }
}

export default DatabasePage;
