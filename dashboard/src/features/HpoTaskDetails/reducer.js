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
import normalize from 'utils/normalize';
import * as types from './actionTypes';

const initialState = {
  notebooks: null, // list of notebooks fetched from the server
  selectedNotebookIds: [], // list of notebooks that have their checkbox checked for possible actions
  notebooksLoading: true,
  searchText: '',
  sortBy: 'NONE',
  hpoTask: null
};

export const HpoTaskDetailsReducer = (state = initialState, action) => {
  switch (action.type) {
    case types.SELECT_NOTEBOOK:
      return {
        ...state,
        selectedNotebookIds: [...state.selectedNotebookIds, action.payload.notebookId]
      };
    case types.UNSELECT_NOTEBOOK:
      return {
        ...state,
        selectedNotebookIds: state.selectedNotebookIds.filter(item => item !== action.payload.notebookId)
      };
    case types.SELECTALL_NOTEBOOKS:
      return {
        ...state,
        selectedNotebookIds: action.payload.notebookIds
      };
    case types.UNSELECTALL_NOTEBOOKS:
      return {
        ...state,
        selectedNotebookIds: []
      };
    case types.GET_NOTEBOOKS_START:
      return {
        ...state,
        notebooksLoading: true
      };
    case types.GET_NOTEBOOKS_SUCCESS: {
      const { byId, allIds } = normalize(action.payload.data, 'id');
      return {
        ...state,
        notebooksLoading: false,
        notebooks: { byId, allIds, count: allIds.length },
        selectedNotebookIds: state.selectedNotebookIds.filter(item => byId[item]),
      };
    }
    case types.GET_HPOTASK_SUCCESS: {
      return {
        ...state,
        hpoTask: action.payload.data
      };
    }
    case types.GET_NOTEBOOKS_FAILURE:
      return {
        ...state,
        notebooksLoading: false,
        notebooks: state.notebooks || {
          byId: {},
          allIds: [],
          count: 0
        }
      };
    case types.SEARCH_TEXT:
      return {
        ...state,
        searchText: action.payload.searchText,
      };
    case types.SORT_BY:
      return {
        ...state,
        sortBy: action.payload.sortBy,
      };
    case types.RESET_STATE:
      return {
        ...initialState
      };
    default:
      return state;
  }
};
