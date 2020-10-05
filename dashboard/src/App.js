import './App.scss';

import React, { Component } from 'react';
//We must use a HashRouter to avoid rewriting the URL during routing
//If we use the BrowserRouter, then we end up with routes like http://server/route
//If we use the HashRouter, we end up with routes like http://server/path/#/route which is what we want.
//See https://github.com/ReactTraining/react-router/blob/master/packages/react-router-dom/docs/api/HashRouter.md
import { HashRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
import { connect } from 'react-redux';
import { Navigation } from './features/Navigation';
import { Dashboard } from './features/Dashboard';
import { Notebooks } from './features/Notebooks';
import { HpoTaskDetails } from './features/HpoTaskDetails';
import { OpenNotifications, ClosedNotifications, createEventStream } from './features/Notifications';
import {Overview} from './features/Overview';
import {TrialsDetail} from './features/Details';
import { Stack } from 'office-ui-fabric-react';
import { COLUMN } from './static/const';
import { EXPERIMENT, TRIALS } from './static/datamodel';

const PrivateRoute = ({ component: Component, data, ...rest }) => (
  <Route
    {...rest}
    render={props => (
      <>
        <div className="app-header">
          <Navigation />
        </div>
        <ClosedNotifications />
        <OpenNotifications />
        <div className="app-page-container">
          <Component {...props} {...data} key={props.match.params.id} />
        </div>
      </>
    )}
  />
);

class App extends Component {
  state = {
    showExpirationModal: false,
    interval: 10, // sendons
    columnList: COLUMN,
    experimentUpdateBroadcast: 0,
    trialsUpdateBroadcast: 0,
    metricGraphMode: 'max'
  };



  async componentDidMount() {
    //this.props.dispatch(checkAuth();
    console.log("start refresher")
    await Promise.all([EXPERIMENT.init(), TRIALS.init()]);
    setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
    setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
    let timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
    setState({ metricGraphMode: (EXPERIMENT.optimizeMode === 'minimize' ? 'min' : 'max') });
  }

  componentDidUpdate(prevProps) {
    if (this.props.authToken && this.props.authToken !== prevProps.authToken) {
      // connect to event listeners
      createEventStream(this.props.authToken);
    }

    if (this.props.isExpired && !prevProps.isExpired) {
      this.onUpdate();
    }
  }

  onUpdate = () => {
    this.setState({ showExpirationModal: true });
  };

  getExpiredRoutes = () => {
    return (
      <>
        <div className="app-header">
          <Navigation />
        </div>
        <div className="app-page-container">
          <Switch>
            <Route exact path="/" component={Dashboard} />
            <Route path="/" render={() => <Redirect to="/" />} />
          </Switch>
        </div>
      </>
    );
  };

  changeColumn = (columnList) => {
      this.setState({ columnList: columnList });
  }

  changeMetricGraphMode = (val) => {
      this.setState({ metricGraphMode: val });
  }
  

  render() {

    return (
      <Router>
        <div className="app-container">
          {
            <Switch>
              <PrivateRoute exact path="/" component={Dashboard}  />
              <PrivateRoute path="/hpotasks/:id" component={HpoTaskDetails}  />
              <PrivateRoute path="/hpotasks" component={Notebooks}  />
              {/* Catches all unknown routes and redirects back to the dashboard */}
              <Redirect from="/" to="/" />
            </Switch>
          }
        </div>
      </Router>
    );
  }

  async refresh() {
    console.log("start refresher - 2")

    const [experimentUpdated, trialsUpdated] = await Promise.all([EXPERIMENT.update(), TRIALS.update()]);
    if (experimentUpdated) {
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
    }
    if (trialsUpdated) {
        this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
    }

    if (['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status)) {
        // experiment finished, refresh once more to ensure consistency
        if (this.state.interval > 0) {
            this.setState({ interval: 0 });
            this.lastRefresh();
        }

    } else if (this.state.interval !== 0) {
        this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
    }
  }



  async lastRefresh(){
      await EXPERIMENT.update();
      await TRIALS.update(true);
      this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
      this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
  }

}

const mapStateToProps = state => {
  return {
    authToken: state.auth.authToken,
    userId: state.auth.userId,
  };
};

export default connect(mapStateToProps)(App);
