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
import React, { Component } from 'react';
import { connect } from 'react-redux';

import { ControlledDataTable } from '../../Common';
import { FilteredNotebooksSelector, SelectedNotebookIdsSelector, NotebooksTableDataSelector } from '../selectors';
import { selectNotebook, unselectNotebook, sortByType } from '../actions';
import { getHeaderList } from '../table-data';

const headers = getHeaderList();

class NotebooksDataTable extends Component {
  toggleRowSelection = (id, checked) => {
    if (checked) {
      this.props.dispatch(selectNotebook(id));
    } else {
      this.props.dispatch(unselectNotebook(id));
    }
  };

  onSortBy = sortType => {
    this.props.dispatch(sortByType(sortType));
  };

  render() {
    const dashIndex = this.props.sortBy ? this.props.sortBy.indexOf('--') : null;
    const sortHeaderKey = dashIndex !== null && dashIndex !== -1 ? this.props.sortBy.substring(0, dashIndex) : null;
    const sortDirection = dashIndex !== null && dashIndex !== -1 ? this.props.sortBy.substring(dashIndex + 2) : 'NONE';
    return (
      <ControlledDataTable
        tableId="notebooks__data-table"
        headers={headers}
        rowIds={this.props.filteredNotebookIds}
        rowsById={this.props.notebooksTableData}
        selectedRows={this.props.selectedNotebookIds}
        sortHeaderKey={sortHeaderKey}
        sortDirection={sortDirection}
        onRowClick={this.props.onRowClick}
        checkboxOnChange={this.toggleRowSelection}
        onSortBy={this.onSortBy}
      />
    );
  }
}

const mapStateToProps = state => {
  return {
    sortBy: state.notebooks.sortBy,
    notebooks: state.notebooks.notebooks,
    filteredNotebookIds: FilteredNotebooksSelector(state),
    selectedNotebookIds: SelectedNotebookIdsSelector(state),
    notebooksTableData: NotebooksTableDataSelector(state),
  };
};

export default connect(mapStateToProps)(NotebooksDataTable);
