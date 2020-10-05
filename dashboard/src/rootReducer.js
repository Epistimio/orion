/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2019                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { combineReducers } from 'redux';
import { NotebooksReducer } from './features/Notebooks';
import { HpoTaskDetailsReducer } from './features/HpoTaskDetails';
import { NotificationsReducer } from './features/Notifications';

export const LOGIN_START = 'auth/LOGIN_START';
export const LOGIN_SUCCESS = 'auth/LOGIN_SUCCESS';
export const LOGIN_FAILURE = 'auth/LOGIN_FAILURE';
export const CHECK_AUTH = 'auth/CHECK_AUTH';
export const LOGOUT_START = 'auth/LOGOUT_START';
export const LOGOUT_SUCCESS = 'auth/LOGOUT_SUCCESS';

const initialState = {
  authToken: null,
  userId: null,
  role: null,
  authLoading: false,
};

export const AuthReducer = (state = initialState, action) => {
  switch (action.type) {
    case CHECK_AUTH:
      return {
        ...state,
        authToken: localStorage.getItem('token'),
        userId: localStorage.getItem('user_id'),
        role: localStorage.getItem('role'),
      }
    case LOGIN_START:
      return {
        ...state,
        authLoading: true,
      }
    case LOGIN_SUCCESS:
      return {
        ...state,
        authToken: action.payload.data.token,
        userId: action.payload.data.user_id,
        role: action.payload.data.role,
        authLoading: false,
      };
    case LOGIN_FAILURE:
    case LOGOUT_SUCCESS:
      return initialState;
    default:
      return state;
  }
};


const appReducers = combineReducers({
  notebooks: NotebooksReducer,
  hpotask: HpoTaskDetailsReducer,
  notifications: NotificationsReducer,
  auth: AuthReducer,
});

const rootReducer = (state, action) => {
  if (action.type === LOGOUT_SUCCESS) {
    // clear all the stores
    // sending undefined causes stores to be set to their initialState
    // eslint-disable-next-line no-param-reassign
    state = undefined;
  }

  return appReducers(state, action);
};

export default rootReducer;
