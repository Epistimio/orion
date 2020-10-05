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

// Gives extra countdown information about the task (ex. 100/200 files).
// Will be displayed in a smaller font-size than the main notification message
const MoreInfoText = React.memo(({ intl, intl_str, is_progress, values }) => {
  if (!intl.messages[intl_str]) {
    // moreInfo intl string doesn't exist, so skip this
    return <div />;
  }

  let decoratedValues = { ...values };
  Object.entries(decoratedValues).forEach(([key, value]) => {
    decoratedValues[key] = !isNaN(value) ? <FormattedNumber value={value} /> : value || <FormattedMessage id="global.unknown" />;
  });

  return (
    <div className={is_progress ? 'task__progress-text' : 'task__msg-text'}>
      <FormattedMessage id={intl_str} values={decoratedValues} />
    </div>
  );
});

export default injectIntl(MoreInfoText);
