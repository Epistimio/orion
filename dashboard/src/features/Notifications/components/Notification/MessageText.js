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
import React from 'react';
import { injectIntl, FormattedMessage, FormattedNumber } from 'react-intl';

// Used to describe the status of the action that is taking place (ex. Auto labeling...)
// or the main sucess/error/warning message for the notification
const MessageText = React.memo(({ intl, is_task, intl_str, values }) => {
  let decoratedValues = values ? { ...values } : {};

  if (is_task) {
    Object.entries(decoratedValues).forEach(([key, value]) => {
      decoratedValues[key] = !isNaN(value) ? (
        <span className="task__subtitle--counter">
          <FormattedNumber value={value} />
        </span>
      ) : (
        <span className="task__subtitle--name">{value}</span>
      );
    });
  } else {
    Object.entries(decoratedValues).forEach(([key, value]) => {
      decoratedValues[key] = !isNaN(value) ? <FormattedNumber value={value} /> : value || <FormattedMessage id="global.unknown" />;
    });
  }

  if (intl.messages[intl_str]) {
    return (
      <div className="bx--toast-notification__subtitle task__subtitle">
        <FormattedMessage id={intl_str} values={decoratedValues} />
      </div>
    );
  }
  return <div />;
});

export default injectIntl(MessageText);
