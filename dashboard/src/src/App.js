import React, { Component } from 'react';
import { ExperimentsWithRouter } from './experiments/Experiments';
import { BenchmarksWithRouter } from './benchmarks/Benchmarks';
import { Route, Switch } from 'react-router-dom';

class App extends Component {
  constructor(props) {
    super(props);
    // Store selected experiment here
    this.state = { page: null };
    this.selectExperiments = this.selectExperiments.bind(this);
    this.selectBenchmarks = this.selectBenchmarks.bind(this);
  }
  render() {
    return (
      <Switch>
        <Route exact path="/" component={ExperimentsWithRouter} />
        <Route exact path="/benchmarks" component={BenchmarksWithRouter} />
        <Route
          exact
          path="/benchmarks/:page"
          component={BenchmarksWithRouter}
        />
        <Route path="/:page" component={ExperimentsWithRouter} />
      </Switch>
    );
  }
  selectExperiments() {
    this.setState({ page: 'experiments' });
  }
  selectBenchmarks() {
    this.setState({ page: 'benchmarks' });
  }
}

export default App;
