/* eslint-disable camelcase */
/* eslint-disable max-len */
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
import { default as baseAxios } from 'axios';
import { notifyByStatus } from 'utils/notify';
import { store } from './index';

import { uploadHandler, clearNotifications } from './features/Notifications';

//note - baseURL can be used to prepend a "base" path to axios URLs, relative to the current location.
//an important bug / note - the react router sets this location, so if the HashRouter is not used, bad
//things happen, resulting in the application being unable to talk to certain URLs
//axios.defaults.baseURL = 'somevalue';
let axios = null;

(function createAxios() {
  const authToken = localStorage.getItem('token');
  axios = baseAxios.create({
    ...(localStorage.getItem('token') ? {
      headers: {
        common: {
          'X-Auth-Token': authToken,
        },
      },
    }
      : {}),
  });

  axios.interceptors.response.use(
    response => {
      
      return response;
    },
    error => {
      return "";
    }
  );
}());

export const setHeader = token => {
  if (token) {
    axios.defaults.headers.common['X-Auth-Token'] = token;
  }
};

export const clearHeader = () => {
  delete axios.defaults.headers.common['X-Auth-Token'];
};

//apis below should be relative URLs (they should not have a leading slash)
/*
 *  AUTHENTICATION
 */



export const getDeviceInfoApi = () => {
  //return axios.get('api/system/device-info');
  return axios.get('http://127.0.0.1:3000/devices');
};

export const getVersionInfoApi = cancelToken => {
  return axios.get('api/version-info', {
    cancelToken,
  });
};

/*
 * HPO
 */
export const getHPOTasksApi = () => {
  return axios.get('/public/hpotasks.json');
};

/*
 * NOTEBOOKS - FETCHING INFORMATION
 */
export const createNotebookApi = name => {
  return axios.post('api/notebooks', {
    name,
  });
};

export const deleteNotebookApi = id => {
  return axios.delete(`api/notebooks/${id}`);
};

/*
 * RESOURCEPLAN - FETCHING INFORMATION
 */
export const getResourcePlanApi = () => {
  return axios.get('http://127.0.0.1:3000/resourceplan');
};

/*
 * APPLICATIONS - FETCHING INFORMATION
 */
export const getApplicationsApi = () => {
  return axios.get('http://127.0.0.1:3000/applications');
};

export const getApplicationApi = applicationId => {
  return axios.get(`http://127.0.0.1:3000/applications/${applicationId}`);
};

/*
 *  MISC ACTIONS
 */
export const fetchImageApi = imagePath => {
  // We need to use axios instead of setting the src on an <img> tag so that
  // an auth token can be sent
  return axios.get(imagePath, {
    responseType: 'blob',
  });
};
