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

// Used to display task progress
const ProgressBar = React.memo(({ notice_type, status, percent_complete }) => {
  if (notice_type === 'info' && percent_complete !== undefined && (status === 'working' || status === 'uploading')) {
    const showAsIndeterminate = isNaN(parseInt(percent_complete));
    return (
      <div className="task__progress-container" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={showAsIndeterminate ? null : percent_complete}>
        {showAsIndeterminate ? <div className="task__progress-bar--indeterminate" /> : <div style={{ width: `${Math.min(percent_complete, 100)}%` }} className="task__progress-bar" />}
      </div>
    );
  }
  return <div />;
});

export default ProgressBar;
