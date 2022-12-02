import React from 'react';
import { DEFAULT_BACKEND } from '../../../utils/queryServer';
import { FeaturedTable } from './FeaturedTable';
import { BackendContext } from '../../BackendContext';
import { TrialsProvider } from './TrialsProvider';
import { ExperimentStatusBar } from '../../components/ExperimentStatusBar';

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
    this.state = { experiment: null, trials: null, trialStatus: null };
    this.onSelectTrialStatus = this.onSelectTrialStatus.bind(this);
  }
  render() {
    if (this.state.experiment === null)
      return 'No trials to display, please select an experiment.';
    if (this.state.trials === null)
      return `Loading trials for experiment "${this.state.experiment}" ...`;
    if (this.state.trials === false)
      return `Unable to load trials for experiment "${this.state.experiment}".`;
    return (
      <div className="database-container">
        <ExperimentStatusBar
          name={this.state.experiment}
          withInfo={true}
          focus={this.state.trialStatus}
          onFocus={this.onSelectTrialStatus}
        />
        <FeaturedTable
          columns={this.state.trials.headers}
          data={
            this.state.trialStatus === null
              ? this.state.trials.trials
              : this.state.trials.trials.filter(
                  trial => trial.status === this.state.trialStatus
                )
          }
          experiment={this.state.experiment}
          trialStatus={this.state.trialStatus}
          nbTrials={this.state.trials.trials.length}
        />
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
  onSelectTrialStatus(trialStatus) {
    console.log(`Db page on focus ${trialStatus}`);
    this.setState({
      trialStatus: this.state.trialStatus === trialStatus ? null : trialStatus,
    });
  }
}

export default DatabasePage;
