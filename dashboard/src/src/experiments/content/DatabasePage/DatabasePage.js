import React from 'react';
import { Backend, DEFAULT_BACKEND } from '../../../utils/queryServer';
import { FeaturedTable } from './FeaturedTable';
import { BackendContext } from '../../BackendContext';
import { Column, Grid, Row } from 'carbon-components-react';
import { flattenObject } from '../../../utils/flattenObject';

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
      /**
       * Map to check whether each param column is sortable.
       * Array params are not considered sortable,
       * except if they contains only 1 element.
       */
      const sortableParamCols = {};
      for (let trialID of trialIndices) {
        const rawTrial = await this.backend.query(
          `trials/${experiment}/${trialID}`
        );
        // Flatten parameters
        const flattenedParameters = flattenObject(
          rawTrial.parameters,
          // Add prefix `params`
          // to prevent collision with existing keys in trial object
          'params'
        );
        // Prepare rendering for array parameters
        for (let key of Object.keys(flattenedParameters)) {
          let sortableCell = true;
          if (Array.isArray(flattenedParameters[key])) {
            if (flattenedParameters[key].length === 1) {
              // Array contains only 1 element.
              // Flatten it and assume element is displayable as-is.
              flattenedParameters[key] = flattenedParameters[key][0];
            } else {
              // Real array with many values.
              // Render it immediately and mark cell as not sortable.
              flattenedParameters[key] = flattenedParameters[
                key
              ].map((value, i) => <div key={i}>{value.toString()}</div>);
              sortableCell = false;
            }
          }
          // Param column is sortable if all its cells are sortable.
          if (sortableParamCols.hasOwnProperty(key)) {
            sortableParamCols[key] = sortableParamCols[key] && sortableCell;
          } else {
            sortableParamCols[key] = sortableCell;
          }
        }
        // Save flattened keys in specific property `paramKeys` for later
        rawTrial.paramKeys = Object.keys(flattenedParameters);
        const trial = { ...rawTrial, ...flattenedParameters };
        // Save statistics as already rendered components.
        trial.statistics = <ObjectToGrid object={trial.statistics} />;
        trials.push(trial);
      }
      // Prepare headers for this experiment using `paramKeys` from first trial
      // We assume paramKeys is the same for all trials
      const paramKeys = trials[0].paramKeys.slice();
      paramKeys.sort();
      const headers = [
        {
          accessorKey: 'id',
          header: 'ID',
          sortingFn: 'text',
          cell: info =>
            info.getValue().length > 7 ? (
              <span title={info.getValue()}>
                {info.getValue().substr(0, 7)}...
              </span>
            ) : (
              info.getValue()
            ),
        },
        {
          // Grouped parameters columns
          header: 'Parameters',
          columns: paramKeys.map(k => {
            const p = { accessorFn: r => r[k], header: k.substr(7) };
            if (!sortableParamCols[k]) {
              // column not sortable
              p.cell = props => props.getValue();
              p.enableSorting = false;
            }
            return p;
          }),
        },
        {
          accessorKey: 'submitTime',
          header: 'Submit time',
        },
        {
          accessorKey: 'startTime',
          header: 'Start time',
        },
        {
          accessorKey: 'endTime',
          header: 'End time',
        },
        {
          accessorKey: 'objective',
          header: 'Objective',
        },
        {
          // not sortable
          accessorKey: 'statistics',
          header: 'Statistics',
          cell: props => props.getValue(),
          enableSorting: false,
        },
      ];
      this.trials[experiment] = {
        headers: headers,
        trials: trials,
      };
    }
    return this.trials[experiment];
  }
}

/**
 * Singleton to provide experiment trials.
 * @type {TrialsProvider}
 */
const TRIALS_PROVIDER = new TrialsProvider(DEFAULT_BACKEND);

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
            <FeaturedTable
              columns={this.state.trials.headers}
              data={this.state.trials.trials}
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
