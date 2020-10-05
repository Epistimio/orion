/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2019, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { createSelector } from 'reselect';

// pull out the pieces of the state that we need inside the selectors
const notificationsById = state => state.notifications.notifications.byId;
const closedIds = state => state.notifications.notifications.closedIds;

/*
 * This selector is used to get a boolean value of true if there is
 * at least one active notification in the closed notification panel
 */
export const HasActiveClosedSelector = createSelector(notificationsById, closedIds, (notificationsById, closedIds) => {
  if (!closedIds.length) {
    return false;
  }
  return closedIds.some(msgId => {
    const msgData = notificationsById[msgId];
    return ['uploading', 'waiting', 'starting', 'working'].includes(msgData.status);
  });
});
