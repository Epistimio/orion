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
import classnames from 'classnames';
import { FormattedMessage } from 'react-intl';
import { InlineLoading } from 'carbon-components-react';

import Notification from './Notification/Notification';
import NotificationDeleteAllModal from './NotificationDeleteAllModal';

import processBackgroundTasks from '../utilities/processBackgroundTasks';
import { getBGTasks, clearNotifications } from '../actions';

/**
 * This area is used to view closed messages that have had their main app notificatons
 * removed (either by clicking the 'x' button or having the countdown timer run out).
 * Only currently active tasks and messages from this user session (since the last app-wide refresh)
 * or stored background tasks will be displayed.
 */

class ClosedNotifications extends Component {
  state = {
    loading: false,
    deleteAllNotificationsModalOpen: false,
  };

  componentDidUpdate(prevProps) {
    // fetch BG Task updates when the user opens the notification center
    // TODO there should be a better time to do this, not sure when though
    if (this.props.notificationBarOpen && this.props.notificationBarOpen !== prevProps.notificationBarOpen) {
      this.onUpdate();
    }
  }

  onUpdate = () => {
    this.setState({ loading: true });
    this.props.getBGTasks().then(response => {
      if (response && response.data && response.data.task_list) {
        // TODO find a way to update these all at once instead of one at a time
        response.data.task_list.forEach(task => processBackgroundTasks(task));
        this.setState({ loading: false });
      }
    });
  };

  sortMessages() {
    const { notifications } = this.props;

    let nonActiveMsgs = [];
    let activeMsgs = notifications.closedIds.filter(msgId => {
      const msg = notifications.byId[msgId];
      if (msg && msg.is_task && ['uploading', 'waiting', 'starting', 'working'].includes(msg.status)) {
        return true;
      }
      nonActiveMsgs.push(msgId);
    });

    const sortMessages = sortParam => (msgId1, msgId2) => {
      let msg1 = notifications.byId[msgId1];
      let msg2 = notifications.byId[msgId2];
      return msg2[sortParam] - msg1[sortParam];
    };
    activeMsgs.sort(sortMessages('created_at'));
    nonActiveMsgs.sort(sortMessages('created_at'));

    return {
      sortedIdList: [...activeMsgs, ...nonActiveMsgs],
      activeTaskCount: activeMsgs.length,
    };
  }

  render() {
    if (!this.props.notificationBarOpen) {
      return <div />;
    }

    const { sortedIdList, activeTaskCount } = this.sortMessages();
    const notificationsClassNames = classnames({
      'notification-overlay': this.state.deleteAllNotificationsModalOpen,
    });
    const notificationClearAllClassNames = classnames({
      'notification-bar__clear-all': true,
      'notification-bar__clear-all--disable': sortedIdList.length === 0,
    });

    return (
      <div className={notificationsClassNames}>
        <div className="notification-bar">
          <div className="notification-bar__recent-title">
            <div className="notification-bar__recent-active-section">
              <FormattedMessage id="notification.recent" />
              <span className="notification-bar__recent-active-separator">&middot;</span>
              <span className="notification-bar__active-count">
                <FormattedMessage id="notification.activeCount" values={{ numActive: activeTaskCount || 0 }} />
              </span>
            </div>
            <div
              className={notificationClearAllClassNames}
              tabIndex={0}
              role="button"
              onClick={() => sortedIdList.length > 0 && this.setState({ deleteAllNotificationsModalOpen: true })}
              onKeyPress={() => sortedIdList.length > 0 && this.setState({ deleteAllNotificationsModalOpen: true })}>
              <FormattedMessage id="notification.deleteAll" />
            </div>
          </div>
          <div className="notification-bar__content">
            {sortedIdList.map(msgId => (
              <Notification key={msgId} msgId={msgId} hideCloseButton />
            ))}
            {this.state.loading && <InlineLoading />}
          </div>
          {this.state.deleteAllNotificationsModalOpen && (
            <NotificationDeleteAllModal
              open={this.state.deleteAllNotificationsModalOpen}
              onClose={() => this.setState({ deleteAllNotificationsModalOpen: false })}
              notifications={this.props.notifications}
              clearNotifications={this.props.clearNotifications}
              getBGTasks={this.props.getBGTasks}
            />
          )}
        </div>
      </div>
    );
  }
}

const mapStateToProps = state => {
  return {
    notificationBarOpen: state.notifications.notificationBarOpen,
    notifications: state.notifications.notifications,
  };
};

const mapDispatchToProps = {
  getBGTasks,
  clearNotifications,
};

export default connect(mapStateToProps, mapDispatchToProps)(ClosedNotifications);
