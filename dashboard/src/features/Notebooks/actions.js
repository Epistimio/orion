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

export const loadHPOTasksData= (dispatch) => {
  Promise.all([
    dispatch(getHPOTasks()).catch(() => { })
  ]).then((responses) => {
    return Promise.resolve(responses);
  });
};


const formatRowData = (data) => {
  if (data) {
    for(let i=0;i<data.length;i++){
      let dataObj = data[i];
      dataObj["params_authorName"] = dataObj.params.authorName ;
      dataObj["params_experimentName"] = dataObj.params.experimentName ;
      dataObj["params_trialConcurrency"] = dataObj.params.trialConcurrency ;
      dataObj["params_maxExecDuration"] = dataObj.params.maxExecDuration ;
      dataObj["params_maxTrialNum"] = dataObj.params.maxTrialNum ;
      dataObj["params_searchSpace"] = dataObj.params.searchSpace ;
      dataObj["params_trainingServicePlatform"] = dataObj.params.trainingServicePlatform ;
      dataObj["params_versionCheck"] = dataObj.params.versionCheck ;
      dataObj["params_tuner_builtinTunerName"] = dataObj.params.tuner.builtinTunerName ;
      dataObj["params_tuner_classArgs"] = JSON.stringify(dataObj.params.tuner.classArgs) ;
      dataObj["params_tuner_checkpointDir"] = dataObj.params.tuner.checkpointDir ;
    }
  }
}

export const getHPOTasks = () => {
  return (dispatch) => {
    dispatch({ type: types.GET_NOTEBOOKS_START });
    return getHPOTasksApi().then(response => {
      let data = response.data;
      formatRowData(data);
      dispatch({
        type: types.GET_NOTEBOOKS_SUCCESS,
        payload: { data: data }
      });
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
