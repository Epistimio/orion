import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { Backend } from '../../../utils/queryServer';
import { BackendContext } from '../../BackendContext';

import {
  SideNav,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListBody,
  StructuredListRow,
  StructuredListInput,
  StructuredListCell,
  Search,
} from 'carbon-components-react';

export class ExperimentNavBar extends React.Component {
  _isMounted = false;
  static contextType = BackendContext;
  constructor(props) {
    // prop: onSelectExperiment: function(experiment)
    super(props);
    this.state = {
      experiments: null,
      search: '',
    };
    this.onSearch = this.onSearch.bind(this);
    this.onSwitchSelect = this.onSwitchSelect.bind(this);
  }
  render() {
    return (
      <SideNav
        className="experiment-navbar"
        isFixedNav
        expanded={true}
        isChildOfHeader={false}
        aria-label="Side navigation">
        <div className="experiments-wrapper">
          <StructuredListWrapper className="experiments-list" selection>
            <StructuredListHead>
              <StructuredListRow head>
                <StructuredListCell className="experiment-cell" head>
                  Experiment
                </StructuredListCell>
                <StructuredListCell head>Status</StructuredListCell>
              </StructuredListRow>
            </StructuredListHead>
            <StructuredListBody>
              {this.renderExperimentsList()}
            </StructuredListBody>
          </StructuredListWrapper>
        </div>
        <Search
          placeholder="Search experiment"
          labelText="Search experiment"
          onChange={this.onSearch}
        />
      </SideNav>
    );
  }
  renderExperimentsList() {
    if (this.state.experiments === null)
      return this.renderMessageRow('Loading experiments ...');
    if (!this.state.experiments.length)
      return this.renderMessageRow('No experiment available');
    // Apply search.
    let experiments;
    if (this.state.search.length) {
      // String to search
      experiments = this.state.experiments.filter(
        experiment => experiment.toLowerCase().indexOf(this.state.search) >= 0
      );
      if (!experiments.length)
        return this.renderMessageRow('No matching experiment');
    } else {
      // No string to search, display all experiments
      experiments = this.state.experiments;
    }
    return experiments.map(experiment => (
      <StructuredListRow
        label
        key={`row-${experiment}`}
        onClick={event =>
          this.onSwitchSelect(
            event,
            experiment,
            `select-experiment-${experiment}`
          )
        }
        {...(this.context.experiment === experiment
          ? {
              className: 'selected-experiment-row',
              title: `unselect experiment '${experiment}'`,
            }
          : {})}>
        <StructuredListInput
          id={`select-experiment-${experiment}`}
          value={`row-${experiment}`}
          title={`row-${experiment}`}
          name="select-experiment"
          onChange={() => this.props.onSelectExperiment(experiment)}
        />
        <StructuredListCell className="experiment-cell">
          <span title={experiment}>{experiment}</span>
        </StructuredListCell>
        <StructuredListCell>
          <ProgressBar>
            <ProgressBar variant="success" now={35} key={1} />
            <ProgressBar variant="warning" now={20} key={2} />
            <ProgressBar variant="danger" now={10} key={3} />
            <ProgressBar variant="info" now={15} key={4} />
          </ProgressBar>
        </StructuredListCell>
      </StructuredListRow>
    ));
  }
  renderMessageRow(message) {
    return (
      <StructuredListRow>
        <StructuredListCell>{message}</StructuredListCell>
        <StructuredListCell />
      </StructuredListRow>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    const backend = new Backend(this.context.address);
    backend
      .query('experiments')
      .then(results => {
        const experiments = results.map(experiment => experiment.name);
        experiments.sort();
        if (this._isMounted) {
          this.setState({ experiments });
        }
      })
      .catch(error => {
        if (this._isMounted) {
          this.setState({ experiments: [] });
        }
      });
  }
  componentWillUnmount() {
    this._isMounted = false;
  }

  onSearch(event) {
    this.setState({ search: (event.target.value || '').toLowerCase() });
  }
  onSwitchSelect(event, experiment, inputID) {
    // Prevent default behavior, as we entirely handle click here.
    event.preventDefault();
    const toBeSelected = this.context.experiment !== experiment;
    document.getElementById(inputID).checked = toBeSelected;
    this.props.onSelectExperiment(toBeSelected ? experiment : null);
  }
}

export default ExperimentNavBar;
