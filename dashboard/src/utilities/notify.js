/* eslint-disable no-param-reassign */
/* eslint-disable consistent-return */
/* eslint-disable camelcase */
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
import { store } from '../index';
import { createNotification } from '../features/Notifications';

export const notify = (intl_id, msgData = {}) => {
  if (!msgData.notice_type) {
    // see if we can pull the type out of the msg title or caption strings; otherwise default to error
    const typeOptions = ['info', 'success', 'error', 'warning'];
    const intlType = (intl_id || '').split('.')[0];
    msgData.notice_type = typeOptions.includes(intlType) ? intlType : 'error';
  }

  if (!msgData.created_at) {
    msgData.created_at = Date.now();
  }
  if (!msgData.updated_at) {
    msgData.updated_at = Date.now();
  }

  store.dispatch(createNotification({ intl_id, ...msgData }));
};

const notify403 = () => {
  // user is trying to access another user's data
  notify('error.403');
};

const notify503 = errorResp => {
  // the instance is expired or the server is unable to handle the request
  const { config, statusText, isExpired } = errorResp;
  if (isExpired) {
    notify('error.trial.expired', {
      values: { url: config.url },
    });
  } else {
    notify('error.503', {
      values: {
        url: config.url,
        statusText: statusText || 'Service Unavailable',
      },
    });
  }
};

export const notifyByStatus = (statusCode, errorResp) => {
  switch (statusCode) {
    case 403:
      return notify403();
    case 503:
      return notify503(errorResp);
    default:
      return undefined;
  }
};
