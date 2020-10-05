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
import React from 'react';
import PropTypes from 'prop-types';
import { useSelector } from 'react-redux';

import { Button } from 'carbon-components-react';
import { Close16 } from '@carbon/icons-react';

import Title from './Title';
import MessageText from './MessageText';
import MoreInfoText from './MoreInfoText';
import ProgressBar from './ProgressBar';
import ErrorText from './ErrorText';
import TimeStamp from './TimeStamp';

/**
 * These are notifications used for both task and toast notifications.
 *
 * Possible msgData values are:
 * notice_type - Required. How the notification is displayed. Possible values include:
 *               "info" (default), "error", "warning", or "success"
 * values - Optional. An object of values that need to be subbed into intl strings.
 *          This object is included in the Title, MessageText, and MoreInfoText components.
 * error - Optional. A normal error object that contains a fault message at error.response.data.fault,
 *         or a string to be used directly as the error message.
 * sub_msg - Optional. Appended to the MessageText intl string created from the intl_id value
 *           to get a more specific message string.
 * sub_more_info - Optional. Appended to the MoreInfoText intl string created from the intl_id value
 *                 to get a more specific more info string.
 * created_at - Optional. Date the notification was created. Should be in millisecond format.
 * resource_str - Optional. If not using a default resource_str, pass in another value.
 * resource_id - Optional. Can be used along with resource_str to create title links for non-task items.
 * resource_name - Optional. If the resource_name being used in the title is known in advance, we can avoid a look-up.
 *
 * Additional msgData values for task notifications are:
 * event_type - Required. Type of task. If this isn't supplied, this isn't considered a task.
 * status - Required. The status of the task, possible values include:
 *          "uploading", "waiting", "starting", "working", "completed", and "failed"
 * resource_id - Optional. The ID of the resource when using a title link.
 * percent_complete - Optional. If showing a non-indeterminate progress bar, needed to show the progress value
 * show_progress_bar - Optional. If true, a progress bar is shown.
 *
 *
 * If you need to create a notification that doesn't fit into this mold, you get pass in
 * titleChildren, messageChildren, moreInfoChildren, or errorChildren properties directly
 */

const typeClassMap = {
  info: 'bx--toast-notification--info',
  success: 'bx--toast-notification--success',
  error: 'bx--toast-notification--error',
  warning: 'bx--toast-notification--warning',
};

const Notification = props => {
  const msgData = useSelector(state => state.notifications.notifications.byId[props.msgId]);

  if (!msgData) {
    // we weren't able to find this msg data
    return <div />;
  }

  const is_task = !!msgData.event_type;

  const createIntlStr = str_type => {
    if (str_type === 'title') {
      return `${is_task ? 'task.' : ''}${msgData.intl_id}.title`;
    }
    // str_type is either "msg" or "moreInfo"
    const sub_str = str_type === 'msg' ? msgData.sub_msg : msgData.sub_more_info;
    const intl_id = ['uploading', 'waiting'].includes(msgData.status) ? msgData.status : msgData.intl_id;
    return `${is_task ? 'task.' + msgData.notice_type + '.' : ''}${intl_id}.${str_type}${sub_str ? '.' + sub_str : ''}`;
  };

  return (
    <div data-notification aria-live="polite" className={`bx--toast-notification bx--toast-notification--low-contrast ${typeClassMap[msgData.notice_type] || typeClassMap.info}`}>
      <div className="bx--toast-notification__details">
        {/* title of the notification; will be plain text or link to resource */}
        {msgData.titleChildren ? (
          msgData.titleChildren
        ) : (
          <Title
            is_task={is_task}
            notice_type={msgData.notice_type}
            intl_str={createIntlStr('title')}
            resource_id={msgData.resource_id}
            task_id={msgData.task_id}
            resource_str={msgData.resource_str}
            resource_name={msgData.resource_name}
            event_type={msgData.event_type}
            value={msgData.values}
          />
        )}

        <div className="task__text-container">
          {/* describes the main point of the notification;
            action taking place or general status message */}
          {msgData.messageChildren ? msgData.messageChildren : <MessageText is_task={is_task} intl_str={createIntlStr('msg')} values={msgData.values} />}

          {/* gives extra, non-essential information about the notification;
            some examples include a countdown (ex. 100/200 files)
            or more info about the success/error/warning message (ex. 21 files imported) */}
          {msgData.moreInfoChildren ? msgData.moreInfoChildren : <MoreInfoText intl_str={createIntlStr('moreInfo')} values={msgData.values} is_progress={msgData.sub_more_info === 'progress'} />}
        </div>

        {/* progress bar showing the task status - will either be at a specific value or indeterminate */}
        {is_task && msgData.show_progress_bar && <ProgressBar notice_type={msgData.notice_type} status={msgData.status} percent_complete={msgData.percent_complete} />}

        {/* show error information if we have it */}
        {msgData.notice_type === 'error' && (msgData.errorChildren ? msgData.errorChildren : <ErrorText error={msgData.error} />)}

        {/* add the date stamp info if it's available */}
        <TimeStamp created_at={msgData.created_at} />
      </div>

      {!props.hideCloseButton && (
        <Button
          data-notification-btn
          className="bx--toast-notification__close-button task__close-button"
          onClick={() => props.closeToast && props.closeToast(props.msgId, msgData.status === 'waiting')}
          kind="ghost"
          size="field"
          type="button"
          renderIcon={Close16}
          iconDescription="Close notification"
        />
      )}
    </div>
  );
};

Notification.propTypes = {
  // id of the message; used to pull msg data out of redux store
  msgId: PropTypes.string.isRequired,
  // function called when the close button is clicked
  closeToast: PropTypes.func,
  // if true, the close button will not be shown
  hideCloseButton: PropTypes.bool,
};

export default Notification;
