/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2020                                           */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import '../styles.scss';

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage, FormattedNumber } from 'react-intl';
import { Loading } from 'carbon-components-react';
import { PageHeader } from '../../PageHeader';
import * as actions from '../actions';
import 'carbon-components/css/carbon-components.min.css';
import { Tabs, Tab, Tooltip } from 'carbon-components-react';
import { Overview } from '../../Overview';
import { TrialsDetail} from '../../Details';

class HpoTaskDetails extends Component {

  constructor(props) {
    super(props);
    this.state = {
      hpoTaskId: props.match.params.id,
    };
  }

  componentDidMount() {
    this.fetchHpoTaskData(true);
  }

  fetchHpoTaskData = (forceFetch) => {
    if (forceFetch) {
      this.props.dispatch(actions.resetState());
      this.props.loadHPOTasksData(this.state.hpoTaskId);
    } else if (!this.props.notebooks) {
      this.props.loadHPOTasksData(this.state.hpoTaskId);
    }
  };

  onClickBackButton = () => {
    const backPath = "/hpotasks";
    this.props.history.push({
      pathname: backPath
    });
  };

  render() {
    const { hpoTask } = this.props;
    return (
      <div className="application-details">
        <div className="application-details__main">
          <PageHeader
            className="pageHeader-details__header"
            showBackButton
            onClickBackButton={this.onClickBackButton}
            title={"HPO Task / " + this.state.hpoTaskId}
            >
            <div className="application-details__subtitle">
              {hpoTask ? (
                <div>
                <span className="application-details__subtitle_label1">Name: </span>{hpoTask.params.experimentName}
                <span className="application-details__subtitle_label">Author: </span> {hpoTask.params.authorName}
                <span className="application-details__subtitle_label">Revision: </span> {hpoTask.revision}
              </div>
              ) : <div/>}
            </div>
          </PageHeader>

          <div className="body-scroll">

            {hpoTask ? (
              <div className="body-container">

                <div style={{ width: '100%' }}>
                  <Tabs type="container">
                    <Tab
                      href="#"
                      id="tab-1"
                      label="Overview">
                      <div className="some-content">
                        <Overview hpoTask={hpoTask}/>
                      </div>
                    </Tab>
                    <Tab
                      href="#"
                      id="tab-2"
                      label="Details">
                      <div className="some-content">
                        <TrialsDetail/>
                      </div>
                    </Tab>
                  </Tabs>
                </div>

              </div>
            ) : <Loading />}
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    hpoTask: state.hpotask.hpoTask
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    loadHPOTasksData : (hpoTaskId) => actions.loadHPOTasksData(dispatch, hpoTaskId),
    dispatch : dispatch
  }
};
export default injectIntl(connect(mapStateToProps, mapDispatchToProps)(HpoTaskDetails));

//export default connect(mapStateToProps, mapDispatchToProps)(Notebooks);