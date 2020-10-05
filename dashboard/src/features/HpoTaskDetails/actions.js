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
import { getHPOTasksApi } from 'api';
import { notify } from 'utils/notify';
import * as types from './actionTypes';

export const loadHPOTasksData= (dispatch, hpoTaskId) => {
  Promise.all([
    dispatch(getHPOTask(hpoTaskId)).catch(() => { })
  ]).then((responses) => {
    return Promise.resolve(responses);
  });
};

export const getHPOTask = (hpoTaskId) => {
  return (dispatch) => {
    dispatch({ type: types.GET_NOTEBOOKS_START });
    return getHPOTasksApi().then(response => {
      let data;
      for(let i=0;i<response.data.length;i++){
        if(response.data[i].id==hpoTaskId){
          data = response.data[i];
          break;
        }
      }

      if(data){
        dispatch({
          type: types.GET_HPOTASK_SUCCESS,
          payload: { data: data }
        });
      }

      return Promise.resolve(response.data);
    }).catch(error => {
      if (error) {
        notify('error.notebooks.get', { error });
      }
      dispatch( {type: types.GET_NOTEBOOKS_FAILURE, payload: { error }} );
      return Promise.reject(error);
    });
  };
};

export const resetState = () => ({
  type: types.RESET_STATE
});

export const selectNotebook = (id) => ({
  type: types.SELECT_NOTEBOOK,
  payload: {
    notebookId: id
  }
});

export const unselectNotebook = (id) => ({
  type: types.UNSELECT_NOTEBOOK,
  payload: {
    notebookId: id
  }
});

export const selectAllNotebooks = (notebookIds) => ({
  type: types.SELECTALL_NOTEBOOKS,
  payload: {
    notebookIds
  }
});

export const unselectAllNotebooks = () => ({
  type: types.UNSELECTALL_NOTEBOOKS
});

export const searchText = (query) => ({
  type: types.SEARCH_TEXT,
  payload: {
    searchText: query
  }
});

export const sortByType = (type) => ({
  type: types.SORT_BY,
  payload: {
    sortBy: type
  }
});
