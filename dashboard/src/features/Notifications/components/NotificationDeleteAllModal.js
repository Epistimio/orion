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
import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { injectIntl, FormattedMessage } from 'react-intl';
import { Checkbox, Tooltip } from 'carbon-components-react';

import { deleteBGTasksApi } from 'api';
import { notify } from 'utils/notify';

import { ActionModal } from '../../Common';
import processBackgroundTasks from '../utilities/processBackgroundTasks';

class NotificationDeleteAllModal extends Component {
  state = {
    submitting: false,
    success: false,
    includeRunningTasks: false,
  };

  onChangeCheckbox = checked => {
    this.setState({ includeRunningTasks: checked });
  };

  onSubmit = () => {
    const { notifications } = this.props;
    let promises = [];

    // toggle the inline-loading icon on the submit button
    this.setState({
      submitting: true,
    });

    // if include running is false, will only delete completed bg tasks; if true,
    // will delete all bg tasks. non-task toast notifications are deleted regardless
    const deleteIds = notifications.closedIds.filter(msgId => {
      const msg = notifications.byId[msgId];
      if (!msg.is_task) {
        // automatically include toast notifications from this session
        return true;
      } if (!this.state.includeRunningTasks) {
        return !['starting', 'working', 'uploading'].includes(msg.status);
      }
      // if the include all running tasks is checked, return all non-uploading messages
      return !['uploading'].includes(msg.status);
    });
    this.props.clearNotifications(deleteIds);

    if (this.state.includeRunningTasks) {
      // If we are including running tasks, we need to make some extra API calls.
      // The query string only takes one status at a time, so we have to
      // make calls to delete "working" and "starting" separately.
      promises.push(
        deleteBGTasksApi('status=working').catch(error => {
          notify('error.bgTask.delete', {
            sub_msg: 'working',
            error,
          });
        })
      );

      promises.push(
        deleteBGTasksApi('status=starting').catch(error => {
          notify('error.bgTask.delete', {
            sub_msg: 'starting',
            error,
          });
        })
      );
    }
    promises.push(
      deleteBGTasksApi().catch(error => {
        notify('error.bgTask.delete', { error });
        this.setState({
          submitting: false,
        });
        this.props.onClose();
      })
    );

    Promise.all(promises).then(() => {
      this.props.getBGTasks().then(response => {
        if (response && response.data && response.data.task_list) {
          response.data.task_list.forEach(task => processBackgroundTasks(task));
        }
      });
      // this will convert the inline-loading icon to a checkbox, which will display
      // for one second and then call the onSuccess function to close the modal
      this.setState({
        submitting: false,
        success: true,
      });
    });
  };

  render() {
    const { intl, open } = this.props;

    return (
      <ActionModal
        open={open}
        danger
        className="notifications-bar__delete-modal"
        modalHeading={intl.formatMessage({ id: 'modal.deleteNotificationsHeader' })}
        primaryButtonText={intl.formatMessage({ id: 'btn.delete' })}
        secondaryButtonText={intl.formatMessage({ id: 'btn.cancel' })}
        onRequestSubmit={this.onSubmit}
        onRequestClose={this.props.onClose}
        submitting={this.state.submitting}
        submittingText={intl.formatMessage({ id: 'btn.deleting' })}
        success={this.state.success}
        onSuccess={this.props.onClose}>
        <FormattedMessage id="modal.deleteNotificationsMessage" />
        <div className="notifications-bar__delete-modal--checkbox">
          <Checkbox
            id="delete-running-notications-checkbox"
            labelText={intl.formatMessage({ id: 'modal.deleteNotificationsTasks' })}
            checked={this.state.includeRunningTasks}
            onChange={this.onChangeCheckbox}
          />
          <Tooltip triggerText="" direction="top" iconDescription="">
            <p>
              <FormattedMessage id="modal.deleteNotificationsTasks.tooltip" />
            </p>
          </Tooltip>
        </div>
      </ActionModal>
    );
  }
}

// Validate the correct prop type is being passed to the component
NotificationDeleteAllModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  notifications: PropTypes.object.isRequired,
  clearNotifications: PropTypes.func,
  getBGTasks: PropTypes.func,
};

export default injectIntl(NotificationDeleteAllModal);
