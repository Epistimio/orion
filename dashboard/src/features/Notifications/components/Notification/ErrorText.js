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

// Used to display error messages from the backend
const ErrorText = React.memo(({ error }) => {
  let fault;

  if (!error) {
    return <div />;
  }

  if (typeof error === 'string') {
    // just use the error string directly
    fault = error;
  } else if (error && error.response && error.response.data) {
    if (error.response.data.fault) {
      // try to pull the fault message out of the error object
      fault = error.response.data.fault;
    } else if (error.response.data.status || error.response.data.more_info) {
      // if there isn't a fault message, there might be more information in
      // the status/more_info fields (Ex. 500 errors have info here)
      fault = error.response.data.status || '';
      fault += fault !== '' && error.response.data.more_info ? ' - ' : '';
      fault += error.response.data.more_info || '';
    }
  } else if (error && !error.response && error.message && typeof error.message === 'string') {
    // this is most likely a built-in http error response (ex. Network Error)
    fault = error.message;
  }

  return <div className="bx--toast-notification__subtitle task__error">{fault}</div>;
});

export default ErrorText;
