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
import '../styles.scss';

import React, { Component } from 'react';
import { connect } from 'react-redux';
import Animate from 'rc-animate';

import Notification from './Notification/Notification';
import NoticeTimer from './NoticeTimer';

import { moveNotification, clearNotifications } from '../actions';

const defaultDuration = {
  'info': 4,
  'success': 4,
  'error': 5,
  'warning': 4,
};

/**
 * This handles toast notifications that show on the main app page. Unlike the notification bar,
 * these notifications will have countdown timers that will cause them to automatically close
 * after a certain amount of time. Once the toast notifications are closed, the messages will appear
 * in the "Recent" section of the notification bar.
 */

class OpenNotifications extends Component {
  createNotification = msgId => {
    const msg = this.props.notifications.byId[msgId];
    // for tasks that are still in progress, the notice shouldn't have a countdown
    // for all other notifications, the duration is the amount of seconds before the notice disappears
    const duration = msg.event_type && ['info', 'error'].includes(msg.notice_type) ? null : msg.duration || defaultDuration[msg.notice_type || 'error'];

    if (msg.event_type) {
      // if they are already viewing training or inference progress, we shouldn't display a notification with the same data
      if (this.props.visibleTrainingId && msg.event_type === 'training' && msg.resource_id === this.props.visibleTrainingId) {
        return;
      }
      if (this.props.visibleDepModelId && msg.event_type === 'inferencing' && msg.resource_id === this.props.visibleDepModelId) {
        return;
      }
    }

    return (
      <NoticeTimer key={msgId} onClose={() => this.closeToast(msgId, msg.status === 'waiting')} duration={duration}>
        <Notification msgId={msgId} closeToast={this.closeToast} />
      </NoticeTimer>
    );
  };

  closeToast = (key, discard) => {
    // discard means it shouldn't be moved to the notification center
    // discard is true for "waiting for import" notifications
    if (discard) {
      this.props.dispatch(clearNotifications([key]));
    } else {
      this.props.dispatch(moveNotification(key, 'closed'));
    }
  };

  render() {
    return (
      <div className="app-notice__container">
        <Animate transitionName="app-notice__fade">{this.props.notifications.openIds.map(this.createNotification)}</Animate>
      </div>
    );
  }
}

const mapStateToProps = state => {
  return {
    notifications: state.notifications.notifications,
    visibleTrainingId: state.notifications.visibleTrainingId,
    visibleDepModelId: state.notifications.visibleDepModelId,
  };
};

export default connect(mapStateToProps)(OpenNotifications);
