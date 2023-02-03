import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { Backend } from '../../../utils/queryServer';
import { BackendContext } from '../../BackendContext';
import InfiniteScroll from 'react-infinite-scroller';

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
      message: 'Loading experiments ...',
      filteredExperiments: [],
      renderedExperiments: [],
    };
    this.onSearch = this.onSearch.bind(this);
    this.onSwitchSelect = this.onSwitchSelect.bind(this);
    this.loadMoreExperiments = this.loadMoreExperiments.bind(this);
    this.hasMoreExperimentToLoad = this.hasMoreExperimentToLoad.bind(this);
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
          <InfiniteScroll
            pageStart={0}
            loadMore={page => this.loadMoreExperiments(page)}
            hasMore={this.hasMoreExperimentToLoad()}
            useWindow={false}
            threshold={5}
            initialLoad={true}>
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
                {this.state.message !== null
                  ? this.renderMessageRow(this.state.message)
                  : this.renderExperimentsList(this.state.renderedExperiments)}
              </StructuredListBody>
            </StructuredListWrapper>
          </InfiniteScroll>
        </div>
        <Search
          placeholder="Search experiment"
          labelText="Search experiment"
          onChange={this.onSearch}
        />
      </SideNav>
    );
  }
  renderExperimentsList(experiments) {
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
          this._updateInitialExperiments(experiments);
        }
      })
      .catch(error => {
        if (this._isMounted) {
          this._updateInitialExperiments([]);
        }
      });
  }
  _updateInitialExperiments(experiments) {
    if (!experiments.length) {
      this.setState({
        experiments,
        search: '',
        message: 'No experiment available',
      });
    } else {
      this.setState({
        experiments,
        search: '',
        message: null,
        filteredExperiments: experiments,
      });
    }
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
  onSearch(event) {
    const search = (event.target.value || '').toLowerCase();
    if (this.state.experiments === null || !this.state.experiments.length) {
      this.setState({ search });
    } else if (search.length) {
      // Apply search.
      const experiments = this.state.experiments.filter(
        experiment => experiment.toLowerCase().indexOf(search) >= 0
      );
      if (!experiments.length) {
        this.setState({
          search,
          message: 'No matching experiment',
          filteredExperiments: [],
          renderedExperiments: [],
        });
      } else {
        this.setState({
          search,
          message: null,
          filteredExperiments: experiments,
          renderedExperiments: [],
        });
      }
    } else {
      // No string to search, display all experiments
      this.setState({
        search,
        message: null,
        filteredExperiments: this.state.experiments,
        renderedExperiments: [],
      });
    }
  }
  onSwitchSelect(event, experiment, inputID) {
    // Prevent default behavior, as we entirely handle click here.
    event.preventDefault();
    const toBeSelected = this.context.experiment !== experiment;
    document.getElementById(inputID).checked = toBeSelected;
    this.props.onSelectExperiment(toBeSelected ? experiment : null);
  }
  hasMoreExperimentToLoad() {
    return (
      this.state.renderedExperiments.length <
      this.state.filteredExperiments.length
    );
  }
  loadMoreExperiments(page) {
    console.log(
      `Loading experiment ${this.state.renderedExperiments.length + 1} / ${
        this.state.filteredExperiments.length
      } (scrolling iteration ${page})`
    );
    this.setState({
      renderedExperiments: this.state.filteredExperiments.slice(
        0,
        this.state.renderedExperiments.length + 1
      ),
    });
  }
}

export default ExperimentNavBar;
