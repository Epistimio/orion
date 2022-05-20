import React, { Component } from 'react';
import './app.scss';
import { Content } from 'carbon-components-react';
import BenchmarkNavBar from './components/BenchmarkNavBar';
import BenchmarkVisualizationsPage from './content/BenchmarkVisualizationsPage';
import { BenchmarkStatusPage } from './content/BenchmarkStatusPage/BenchmarkStatusPage';
import { BenchmarkDatabasePage } from './content/BenchmarkDatabasePage/BenchmarkDatabasePage';
import { BenchmarkConfigurationPage } from './content/BenchmarkConfigurationPage/BenchmarkConfigurationPage';
import TutorialHeader from '../experiments/components/TutorialHeader';

import { DEFAULT_BACKEND } from '../utils/queryServer';
import { Backend } from '../utils/queryServer';
import { withRouter } from 'react-router-dom';

class Benchmarks extends Component {
  _isMounted = false;
  constructor(props) {
    super(props);
    // Store selected experiment here
    this.state = {
      benchmarks: null,
      benchmark: null,
      algorithms: null,
      tasks: null,
      assessments: null,
    };
    this.onSelectBenchmark = this.onSelectBenchmark.bind(this);
  }
  render() {
    return (
      <>
        <TutorialHeader dashboard="benchmarks" />
        {this.state.benchmarks === null ? (
          <Content>
            <h4>Loading benchmarks ...</h4>
          </Content>
        ) : this.state.benchmarks.length === 0 ? (
          <Content>
            <h4>No benchmarks available</h4>
          </Content>
        ) : (
          <>
            <BenchmarkNavBar
              benchmarks={this.state.benchmarks}
              benchmark={this.state.benchmark}
              algorithms={this.state.algorithms}
              tasks={this.state.tasks}
              assessments={this.state.assessments}
              onSelectBenchmark={this.onSelectBenchmark}
            />
            <Content>{this.renderPage()}</Content>
          </>
        )}
      </>
    );
  }
  renderPage() {
    switch (this.props.match.params.page || 'visualizations') {
      case 'status':
        return <BenchmarkStatusPage />;
      case 'database':
        return <BenchmarkDatabasePage />;
      case 'configuration':
        return <BenchmarkConfigurationPage />;
      case 'visualizations':
        return (
          <BenchmarkVisualizationsPage
            benchmark={this.state.benchmark}
            algorithms={this.state.algorithms}
            tasks={this.state.tasks}
            assessments={this.state.assessments}
          />
        );
      default:
        break;
    }
  }
  componentDidMount() {
    this._isMounted = true;
    const backend = new Backend(DEFAULT_BACKEND);
    backend
      .query('benchmarks')
      .then(benchmarks => {
        if (this._isMounted) {
          this.setState({ benchmarks });
        }
      })
      .catch(error => {
        console.error(error);
        if (this._isMounted) {
          this.setState({ benchmarks: [] });
        }
      });
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
  onSelectBenchmark(benchmark, algorithms, tasks, assessments) {
    this.setState({ benchmark, algorithms, tasks, assessments });
  }
}

export const BenchmarksWithRouter = withRouter(Benchmarks);
