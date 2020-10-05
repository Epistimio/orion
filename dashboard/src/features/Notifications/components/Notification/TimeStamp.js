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
import { FormattedDate } from 'react-intl';

// Displays a timestamp of when the notification was created
const TimeStamp = React.memo(({ created_at }) => {
  if (!isNaN(parseInt(created_at)) && created_at !== 0) {
    return (
      <div className="bx--toast-notification__caption task__time-stamp">
        <FormattedDate value={new Date(+created_at)} year="numeric" month="short" day="numeric" hour="numeric" minute="numeric" second="numeric" />
      </div>
    );
  }
  return <div />;
});

export default TimeStamp;
