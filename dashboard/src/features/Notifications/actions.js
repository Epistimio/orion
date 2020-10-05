/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { getBGTasksApi } from 'api';
import { notify } from 'utils/notify';
import * as types from './actionTypes';

export const getBGTasks = () => {
  return dispatch => {
    dispatch({ type: types.GET_BGTASKS_START });
    return getBGTasksApi()
      .then(response => {
        dispatch({
          type: types.GET_BGTASKS_SUCCESS,
          payload: { data: response.data },
        });
        return Promise.resolve(response);
      })
      .catch(error => {
        if (error) {
          notify('error.bgTask.get', { error });
        }
        dispatch({
          type: types.GET_BGTASKS_FAILURE,
        });
        return Promise.reject(error);
      });
  };
};

export const createNotification = (msgData, msgId, location) => ({
  // msgId and location are optional
  type: types.CREATE_NOTIFICATION,
  payload: {
    msgId,
    msgData,
    location,
  },
});

export const updateNotification = (msgData, msgId) => ({
  type: types.UPDATE_NOTIFICATION,
  payload: {
    msgId,
    msgData,
  },
});

export const moveNotification = (msgId, toLocation = 'closed') => ({
  type: types.MOVE_NOTIFICATION,
  payload: {
    msgId,
    toLocation,
  },
});

export const clearNotifications = msgIdList => ({
  type: types.CLEAR_NOTIFICATIONS,
  payload: {
    msgIdList,
  },
});

export const setVisibleTraining = modelId => ({
  type: types.SET_VISIBLE_TRAINING,
  payload: {
    modelId,
  },
});

export const setVisibleDeployedModel = deployedModelId => ({
  type: types.SET_VISIBLE_DEPMODEL,
  payload: {
    deployedModelId,
  },
});

export const toggleNotificationCenter = show => ({
  type: types.TOGGLE_NOTIFICATION_CENTER,
  payload: {
    show,
  },
});
