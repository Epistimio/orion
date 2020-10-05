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
import React, { PureComponent } from 'react';
import { connect } from 'react-redux';

import { Toolbar } from '../../Common';
import { SelectedNotebookIdsSelector, FilteredNotebooksSelector } from '../selectors';
import { searchText, selectAllNotebooks, unselectAllNotebooks } from '../actions';

class NotebooksToolbar extends PureComponent {
  constructor(props) {
    super(props);

    this.onOpenDelete = this.props.openModal.bind(this, 'delete');
  }

  getToolbarList = () => {
    return [
      { type: 'select', disabled: this.props.zeroTotalItems, onChange: this.onChangeSelectCheckbox },
      {
        type: 'buttons',
        includeButtons: [
          { type: 'deleteBtn', disabled: (this.props.noneSelected), onClick: this.onOpenDelete },
          { type: 'refresh', onClick: this.onRefresh }
        ]
      },
      { type: 'search', onChange: this.onSearch, defaultValue: this.props.searchText }
    ];
  };

  onSearch = event => {
    // extract and save a "lower case" search text, IF the event has a value.
    // "value" may not be set if the search bar was cleared.
    const query = event.target.value ? event.target.value.toLocaleLowerCase() : '';
    this.props.dispatch(searchText(query));
  };

  onChangeSelectCheckbox = value => {
    // value will either be True/False - if selection was made via checkbox,
    // or it will be the selected item - if selection was made from the dropdown
    if ((value && typeof (value) !== 'object') || (value && value.selectedItem.id === 'all')) {
      this.props.dispatch(selectAllNotebooks(this.props.filteredNotebookIds));
    } else {
      this.props.dispatch(unselectAllNotebooks());
    }
  };

  onRefresh = () => {
    this.props.fetchNotebooks(true);
  };

  render() {
    return (
      <Toolbar
        includeWidgets={this.getToolbarList()}
        selectCheckbox={this.props.selectCheckbox}
        showSearch={this.props.searchText !== ''}
      />
    );
  }
}

const mapStateToProps = state => {
  return {
    searchText: state.notebooks.searchText,
    notebooks: state.notebooks.notebooks,
    selectedNotebookIds: SelectedNotebookIdsSelector(state),
    filteredNotebookIds: FilteredNotebooksSelector(state),
  };
};

export default connect(mapStateToProps)(NotebooksToolbar);
