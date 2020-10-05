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
import normalize from 'utils/normalize';
import * as types from './actionTypes';

const initialState = {
  notificationBarOpen: false, // if the notification overlay is visible or not
  notifications: {
    byId: {},
    allIds: [],
    closedIds: [], // notifications that only show up in the notification bar
    openIds: [], // notifications that are still being shown on the main app screen
  },
  bgTasks: null,
  bgTasksLoading: false,
  key: '1',
  visibleTrainingId: null,
  visibleDepModelId: null,
};

const createNotification = (state, action) => {
  // If there is another message with the exact same task_id, it will be overwritten
  const msgId = action.payload.msgId || state.key;
  const inClosed = action.payload.location === 'closed' || state.notificationBarOpen;

  return {
    ...state,
    key: msgId === state.key ? (+state.key + 1).toString() : state.key,
    notifications: {
      ...state.notifications,
      byId: {
        ...state.notifications.byId,
        [msgId]: action.payload.msgData,
      },
      allIds: [...state.notifications.allIds, msgId],
      ...(!inClosed
        ? {
          openIds: [...state.notifications.openIds, msgId],
        }
        : {}),
      ...(inClosed
        ? {
          closedIds: [...state.notifications.closedIds, msgId],
        }
        : {}),
    },
  };
};

const updateNotification = (state, action) => ({
  ...state,
  notifications: {
    ...state.notifications,
    byId: {
      ...state.notifications.byId,
      [action.payload.msgId]: {
        ...state.notifications.byId[action.payload.msgId],
        ...action.payload.msgData,
      },
    },
  },
});

const moveNotification = (state, action) => {
  return {
    ...state,
    notifications: {
      ...state.notifications,
      openIds: action.payload.toLocation === 'closed' ? state.notifications.openIds.filter(id => id !== action.payload.msgId) : [...state.notifications.openIds, action.payload.msgId],
      closedIds: action.payload.toLocation === 'open' ? state.notifications.closedIds.filter(id => id !== action.payload.msgId) : [...state.notifications.closedIds, action.payload.msgId],
    },
  };
};

const clearNotifications = (state, action) => {
  let updatedById = { ...state.notifications.byId };
  action.payload.msgIdList.forEach(msgId => {
    delete updatedById[msgId];
  });

  return {
    ...state,
    notifications: {
      byId: updatedById,
      allIds: state.notifications.allIds.filter(id => !action.payload.msgIdList.includes(id)),
      closedIds: state.notifications.closedIds.filter(id => !action.payload.msgIdList.includes(id)),
      openIds: state.notifications.openIds.filter(id => !action.payload.msgIdList.includes(id)),
    },
  };
};

const setVisibleTraining = (state, action) => ({
  ...state,
  visibleTrainingId: action.payload.modelId,
});

const setVisibleDeployedModel = (state, action) => ({
  ...state,
  visibleDepModelId: action.payload.deployedModelId,
});

const toggleNotificationCenter = (state, action) => ({
  ...state,
  notificationBarOpen: action.payload.show,
});

const getBGTasksStart = state => ({
  ...state,
  tasksLoading: true,
});

const getBGTasksSuccess = (state, action) => {
  const { byId, allIds, count } = normalize(action.payload.data.task_list, '_id');

  return {
    ...state,
    tasksLoading: false,
    bgTasks: { byId, allIds, count },
  };
};

const getBGTasksFailure = state => ({
  ...state,
  tasksLoading: false,
  bgTasks: state.bgTasks || {
    byId: {},
    allIds: [],
    count: 0,
  },
});

export const NotificationsReducer = (state = initialState, action) => {
  switch (action.type) {
    case types.CREATE_NOTIFICATION:
      return createNotification(state, action);
    case types.UPDATE_NOTIFICATION:
      return updateNotification(state, action);
    case types.MOVE_NOTIFICATION:
      return moveNotification(state, action);
    case types.CLEAR_NOTIFICATIONS:
      return clearNotifications(state, action);
    case types.SET_VISIBLE_TRAINING:
      return setVisibleTraining(state, action);
    case types.SET_VISIBLE_DEPMODEL:
      return setVisibleDeployedModel(state, action);
    case types.TOGGLE_NOTIFICATION_CENTER:
      return toggleNotificationCenter(state, action);
    case types.GET_BGTASKS_START:
      return getBGTasksStart(state);
    case types.GET_BGTASKS_SUCCESS:
      return getBGTasksSuccess(state, action);
    case types.GET_BGTASKS_FAILURE:
      return getBGTasksFailure(state);
    default:
      return state;
  }
};
