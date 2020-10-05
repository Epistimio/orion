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
import { createSelector } from 'reselect';
import { sortIds } from 'utils/sorting';
import { getHeaderList, createSearchSortData } from './table-data';

const notebooksSelector = state => state.notebooks.notebooks;
const sortBySelector = state => state.notebooks.sortBy;
const searchTextSelector = state => state.notebooks.searchText;
const selectedNotebookIdsSelector = state => state.notebooks.selectedNotebookIds;

// Puts together the information needed for searching and sorting
// every time notebooks are re-fetched.
const CreateSearchSortDataSelector = createSelector(
  notebooksSelector,
  // this selector is only called if the notebooks object is changed
  (notebooks) => {
    const { searchData, sortData } = createSearchSortData(notebooks);
    return {
      notebooks,
      searchData,
      sortData
    };
  }
);

// Sorts data according to the given key/direction.
const SortNotebooksSelector = createSelector(
  CreateSearchSortDataSelector,
  sortBySelector,
  // this selector is called if the notebooks object changes OR the sort by value changes
  (notebooksData, sortBy) => {
    const { notebooks, sortData } = notebooksData;
    if (notebooks && notebooks.allIds) {
      if (sortBy && sortBy !== 'NONE') {
        const dashIndex = sortBy.indexOf('--');
        const sortHeaderKey = dashIndex !== -1 ? sortBy.substring(0, dashIndex) : null;
        const sortDirection = dashIndex !== -1 ? sortBy.substring(dashIndex + 2) : 'NONE';
        return sortIds({
          idList: notebooks.allIds,
          objectByIdMap: sortData,
          sortDirection,
          sortHeaderKey,
        });
      }
      return notebooks.allIds;
    }
    return [];
  }
);

// If search text is entered, will search the table data and return those items.
const SearchNotebooksSelector = createSelector(
  CreateSearchSortDataSelector,
  SortNotebooksSelector,
  searchTextSelector,
  // this selector is called if the notebooks list changes, the sort by value changes, OR the search text changes
  (notebooksData, sortedIdList, searchText) => {
    const { searchData } = notebooksData;
    if ((sortedIdList && sortedIdList.length) && (searchText && searchText.length > 0)) {
      // find the data set IDs which match (in lower case) the search text
      let matchText = searchText.toLocaleLowerCase();
      return sortedIdList.filter(item => searchData[item].includes(matchText));
    }
    // there's no searching to do, just return the list of IDs
    return sortedIdList;
  }
);

const getSelectedItems = (filteredNotebookIds, selectedNotebookIds) => {
  const visibleIds = filteredNotebookIds || [];
  const checkedIds = selectedNotebookIds || [];
  return visibleIds.filter(item => checkedIds.includes(item));
};

export const FilteredNotebooksSelector = createSelector(
  SearchNotebooksSelector,
  (searchResults) => {
    return searchResults;
  }
);

export const SelectedNotebookIdsSelector = createSelector(
  //note we intentionally chain together the SelectedNotebookIdsSelector and the FilteredNotebookSelector so that
  //we get a list of items that are:
  // 1. available to the user based on some filter
  // 2. checked (manually or via 'select all')
  // There's a very slight annoyance here - if a user filters, selects all, and then removes the filter, then
  // all selected items are returned. Should this instead be "sticky," so that if I select 3 items, remove the filter,
  //
  FilteredNotebooksSelector,
  selectedNotebookIdsSelector,
  getSelectedItems
);

// Puts together the information needed by the data table every time
// notebooks are re-fetched.
export const NotebooksTableDataSelector = createSelector(
  notebooksSelector,
  (notebooks) => {
    const headers = getHeaderList();
    let rowsById = {};
    notebooks && notebooks.allIds.forEach((notebookId) => {
      let assetList = [];
      const notebook = notebooks.byId[notebookId];

      if (!notebook.name) {
        notebook.name = notebookId; // in case the name does not exist, use the id of the notebook
      }

      //Sum up the total items in the notebook
      notebook.total_items = 0;

      // Construct assets string for the table
      
      notebook.assets = assetList.join(', ');

      rowsById[notebookId] = {
        id: notebookId,
        cells: headers.map((header) => ({
          id: `${notebookId}:${header.key}`,
          value: header.decorator ? header.decorator(notebook[header.key])
            : (notebook[header.key] || (notebook[header.key] === 0 ? notebook[header.key] : '—'))
        }))
      };
    });
    return rowsById;
  }
);
