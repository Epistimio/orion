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

import { processEvent } from './processSSEEvent';

const sendUploadProgress = taskInfo => {
  // taskInfo should include a unique task_id and an event_type;
  // extra info can include a noSSE value (which means no SSEs will come after this event),
  // and any other information that is allowed for an SSE with the same event_type
  processUploadEvent({
    resource_id: null, // default to null, but will be replaced if included in taskInfo
    status: 'uploading',
    ...taskInfo,
  });
};

const sendUploadCompleted = taskInfo => {
  // taskInfo should include a unique task_id and an event_type
  processUploadEvent({
    resource_id: null, // default to null, but will be replaced if included in taskInfo
    status: 'completed',
    progressEvent: { total: 100, loaded: 100 },
    ...taskInfo,
  });
};

const sendUploadFailed = taskInfo => {
  // taskInfo should include a unique task_id and an event_type;
  // can also include an error object in taskInfo to show more info on why it failed
  processUploadEvent({
    resource_id: null, // default to null, but will be replaced if included in taskInfo
    status: 'failed',
    success_count: 0,
    fail_count: 1,
    total_file_count: 1, // default to 1 out of 1 files failed, but will be replaced if included in taskInfo
    progressEvent: { total: 100, loaded: 100 },
    ...taskInfo,
  });
};

const sendUploadCanceled = taskInfo => {
  // TODO add more functionality later when canceled is supported
  processUploadEvent({
    resource_id: null,
    status: 'canceled',
    progressEvent: { total: 100, loaded: 100 },
    ...taskInfo,
  });
};

const processUploadEvent = data => {
  let { status } = data;
  const { loaded, total } = data.progressEvent;

  data.percent_complete = Math.min(Math.floor((loaded * 100) / total), 100);
  data.values = {};

  if (status === 'uploading') {
    if (data.name) {
      data.values.name = data.name;
      data.sub_msg = 'name';
    } else if (data.event_type === 'import_files' && data.total_file_count > 1) {
      data.sub_msg = 'multiple';
    }
  }

  if (data.event_type === 'import_files') {
    // for import files, SSEs are only returned when submitting .zip files.
    // therefore, we have to handle the success message and non-backend error messages ourselves here;
    // the API caller will handle error messages with proper back-end responses
    if (status === 'failed') {
      data.fail_count = data.total_file_count;
      if (data.possible_folder) {
        // the back-end fails without sending an error message if a folder was uploaded
        // so show a good error message here if we think there's a folder (file with a type of "")
        data.sub_more_info = 'folder';
      } else if (data.name) {
        data.sub_more_info = 'name';
        data.values.name = data.name;
      }
    } else if (data.no_sse && status === 'completed') {
      data.success_count = data.total_file_count;
    }
  }

  if (data.status === 'uploading') {
    if (data.percent_complete === 100) {
      // the file has been completed uploaded, now we're just
      // waiting for an SSE to come in to show importing progress
      data.status = 'waiting';
      data.sub_msg = null;
    } else if (!isNaN(parseInt(loaded)) && !isNaN(parseInt(total))) {
      // show uploading progress
      let expLoaded = Math.floor(Math.log(loaded) / Math.log(1000));
      let expTotal = Math.floor(Math.log(total) / Math.log(1000));
      data.values.current_count = `${(loaded / Math.pow(1000, expLoaded)).toFixed(2) * 1} ${['B', 'kB', 'MB', 'GB'][expLoaded]}`;
      data.values.total_count = `${(total / Math.pow(1000, expTotal)).toFixed(2) * 1} ${['B', 'kB', 'MB', 'GB'][expTotal]}`;
      data.sub_more_info = 'progress';
    }
  }

  processEvent({ data });
};

export const uploadHandler = {
  sendUploadProgress,
  sendUploadCompleted,
  sendUploadFailed,
  sendUploadCanceled,
};
