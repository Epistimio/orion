import React, { Component } from 'react';
import { Content } from 'carbon-components-react';
import TutorialHeader from './components/TutorialHeader';
import ExperimentNavBar from './components/ExperimentNavBar';
import LandingPage from './content/LandingPage';
import StatusPage from './content/StatusPage';
import VisualizationsPage from './content/VisualizationsPage';
import DatabasePage from './content/DatabasePage';
import ConfigurationPage from './content/ConfigurationPage';
import { BackendContext } from './BackendContext';
import { DEFAULT_BACKEND } from '../utils/queryServer';
import { withRouter } from 'react-router-dom';

class Experiments extends Component {
  constructor(props) {
    super(props);
    // Store selected experiment here
    this.state = { experiment: null };
    this.onSelectExperiment = this.onSelectExperiment.bind(this);
  }
  render() {
    return (
      <>
        <BackendContext.Provider
          value={{
            address: DEFAULT_BACKEND,
            // Pass selected experiment as React context
            // so that it is available in route components
            experiment: this.state.experiment,
          }}>
          <TutorialHeader dashboard="experiments" />
          <ExperimentNavBar onSelectExperiment={this.onSelectExperiment} />
          <Content>{this.renderPage()}</Content>
        </BackendContext.Provider>
      </>
    );
  }
  renderPage() {
    switch (this.props.match.params.page || 'landing') {
      case 'landing':
        return <LandingPage key={this.state.experiment} />;
      case 'status':
        return <StatusPage key={this.state.experiment} />;
      case 'visualizations':
        return <VisualizationsPage key={this.state.experiment} />;
      case 'database':
        return <DatabasePage key={this.state.experiment} />;
      case 'configuration':
        return <ConfigurationPage key={this.state.experiment} />;
      default:
        break;
    }
  }
  onSelectExperiment(experiment) {
    this.setState({ experiment });
  }
}

export const ExperimentsWithRouter = withRouter(Experiments);
