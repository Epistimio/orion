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
import React, { Component } from 'react';
import PropTypes from 'prop-types';
import classnames from 'classnames';

/**
 * Notice is a wrapper for toast/task notifications. It holds the countdown timer
 * functionality, so it's only used in the app-wide Notifications.js file, and not the
 * NotificationBar.js file which keeps notifications until they are manually closed.
 */

class NoticeTimer extends Component {
  componentDidMount() {
    this.startCloseTimer();
  }

  componentWillUnmount() {
    this.clearCloseTimer();
  }

  componentDidUpdate(prevProps) {
    if (this.props.duration !== prevProps.duration) {
      this.clearCloseTimer();
      this.startCloseTimer();
    }
  }

  close = () => {
    this.clearCloseTimer();
    if (this.props.onClose) {
      this.props.onClose();
    }
  };

  startCloseTimer = () => {
    if (this.props.duration) {
      this.closeTimer = setTimeout(() => {
        this.close();
      }, this.props.duration * 1000);
    }
  };

  clearCloseTimer = () => {
    if (this.closeTimer) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
  };

  render() {
    const classNames = classnames('app-notice__notification', this.props.className);

    return (
      <div className={classNames} onMouseEnter={this.clearCloseTimer} onMouseLeave={this.startCloseTimer}>
        <div className="app-notice__notification-content">{this.props.children}</div>
      </div>
    );
  }
}

NoticeTimer.propTypes = {
  duration: PropTypes.number,
  onClose: PropTypes.func,
  children: PropTypes.any,
};

export default NoticeTimer;
