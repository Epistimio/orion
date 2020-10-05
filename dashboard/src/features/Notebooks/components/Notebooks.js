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

import NotebooksToolbar from './NotebooksToolbar';
import NotebooksDataTable from './NotebooksDataTable';
import { PageHeader } from '../../PageHeader';

import * as actions from '../actions';
import { SelectedNotebookIdsSelector, FilteredNotebooksSelector } from '../selectors';

class Notebooks extends Component {
  state = {
    
  };

  componentDidMount() {
    this.fetchNotebooks(true);
  }

  fetchNotebooks = (forceFetch) => {
    // if we already have notebooks info, only re-fetch it if forceFetch is true
    if (forceFetch) {
      // used when refreshing from the toolbar, first clears out all current notebooks info (and selections and search text)
      this.props.dispatch(actions.resetState());
      this.props.loadHPOTasksData();
    } else if (!this.props.notebooks) {
      this.props.loadHPOTasksData();
    }
  };

  viewNotebooksDetails = (notebookId) => {
    let locationIndex = window.location.href.lastIndexOf('/');
    // only go to details page if the user clicked on a row
    if (window.location.href.substring(locationIndex + 1) === 'notebooks') {
      const notebook = this.props.notebooks && this.props.notebooks.byId[notebookId];
      if (notebook) {
        // open up notebook details through the Router
        this.props.history.push({
          pathname: `/notebooks/${notebookId}`
        });
      }
    }
  };

  openModal = (modalType) => {
    this.setState({
      [modalType + 'ModalOpen']: true
    });
  };

  render() {
    const numSelected = this.props.selectedNotebookIds ? this.props.selectedNotebookIds.length : 0;
    const totalItems = this.props.filteredNotebookIds ? this.props.filteredNotebookIds.length : 0;
    const selectCheckboxValue = numSelected === 0 ? 'none' : numSelected !== totalItems ? 'indeterminate' : 'all';

    return (
      <div className="notebooks">
        <PageHeader
          title={<FormattedMessage id="header.notebooks" />} >
          <div className="page-header__selected-text">
            <FormattedMessage id="header.selected" />
            <div className="page-header__text-items">
              <FormattedNumber value={numSelected} />
              /
              <FormattedNumber value={totalItems} />
            </div>
          </div>
        </PageHeader>

        <NotebooksToolbar
          openModal={this.openModal}
          fetchNotebooks={this.fetchNotebooks}
          selectCheckbox={selectCheckboxValue}
          zeroTotalItems={totalItems === 0}
          noneSelected={numSelected === 0}
          singleSelected={numSelected === 1}
          multiSelected={numSelected > 1}
        />

        <div className="body-scroll">
          <div className="notebooks body-container">
            {this.props.notebooks ? (
              <div>
                
                {this.props.notebooks.count !== 0 && (
                  <div>
                    <NotebooksDataTable onRowClick={this.viewNotebooksDetails} />
                  </div>
                )}
                
              </div>
            ) : (
              <Loading />
            )}
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    notebooks: state.notebooks.notebooks,
    selectedNotebookIds: SelectedNotebookIdsSelector(state),
    filteredNotebookIds: FilteredNotebooksSelector(state)
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    loadHPOTasksData : () => actions.loadHPOTasksData(dispatch),
    dispatch : dispatch
  }
};
export default injectIntl(connect(mapStateToProps, mapDispatchToProps)(Notebooks));

//export default connect(mapStateToProps, mapDispatchToProps)(Notebooks);