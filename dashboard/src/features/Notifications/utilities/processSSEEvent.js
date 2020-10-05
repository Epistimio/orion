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

// uploading, waiting, starting, working, and most training statuses will default to info
const nonInfoStatusTypeMap = {
  failed: 'error',
  aborted: 'error',
  completed: 'success',
  ready: 'success',
  canceled: 'success',
  deleted: 'warning',
};

const dataDefaults = {
  values: {},
  is_task: true,
  sub_msg: null,
  sub_more_info: null,
  percent_complete: 0,
  show_progress_bar: false,
};

const makeFloat = value => {
  // is used to guarantee that numbers are supplied in number (not string) form, or default to 0
  return !isNaN(parseFloat(value)) ? parseFloat(value) : 0;
};

export const processEvent = (msg, type) => {
  // Every SSE and Background Task should at least have this information:
  // task_id or _id
  // status
  // event_type (on bg task, add for sse)
  // timestamp or updated_at
  let event = typeof msg.data === 'object' ? msg.data : JSON.parse(msg.data);

  const data = {
    ...dataDefaults,
    ...event,
    event_type: type || event.event_type,
    task_id: event.task_id || event._id,
    show_progress_bar: (event.status === 'working' || event.status === 'uploading') && event.percent_complete >= 0,
    notice_type: nonInfoStatusTypeMap[event.status] || 'info',
    updated_at: event.is_bgtask ? event.updated_at : event.timestamp || Date.now(),
    ...(event.createdAt ? { created_at: event.createdAt } : {}),
  };

  // run the event data through the individual event type's
  // process function to make sure the data is in the correct format
  const processFunc = processorMap[type || event.event_type];
  const processedData = processFunc ? processFunc(data) : data;
  
};



const processImportFilesMsg = data => {
  /*
   * guaranteed properties:
   * task_id, resource_id, dataset_id, status, total_file_count, timestamp
   *
   * working/completed properties:
   * percent_complete, success_count, fail_count, skip_count
   *
   * status values:
   * “starting", “working", “completed"
   */

  if (makeFloat(data.fail_count) > 0 && data.notice_type === 'success') {
    // convert to a warning message if a file failed to be imported
    data.notice_type = 'warning';
  }
  data.task_id = data.task_id || `import_files-${data.resource_id}-${data.total_file_count}`;

  data.values = {
    success_count: makeFloat(data.success_count), // used by success
    fail_count: makeFloat(data.fail_count), // used by warning and error
    current_count: makeFloat(data.success_count) + makeFloat(data.fail_count) + makeFloat(data.skip_count), // used by info
    total_file_count: makeFloat(data.total_file_count), // used by info
    ...data.values, // include any values from the uploading code
  };

  if (data.status === 'working' && data.values.total_file_count) {
    data.sub_more_info = 'progress'; // needs current_count and total_file_count
  }
  data.intl_id = 'file.import';

  return data;
};


const processImportModelMsg = data => {
  // TODO import models has not been implemented yet, recheck this function when it is
  data.intl_id = 'model.import';

  return data;
};

const processDeployModelMsg = data => {
  /*
   * guaranteed properties:
   * task_id, resource_id, status, timestamp
   *
   * status values:
   * “starting", "ready", "failed"
   */

  if (data.status === 'starting') {
    // ignore "starting" messages since there can be a timing issue where two models are deployed at once
    // and one of the SSE notifications will stay in "starting" forever until a GPU is freed up.
    // see issue #5077 and #5173 for more info
    return;
  }

  if (data.status === 'ready') {
    // ready means completed, so change the value so that it works with the rest of the notification code
    data.status = 'completed';
  }
  data.intl_id = 'model.deploy';

  return data;
};

const processInferencingMsg = data => {
  /*
   * guaranteed properties:
   * task_id, resource_id, status, inference_type ("Action" or "Object"), timestamp
   * (modified below so the task_id is the inference_id, resource_id is
   * the deployed model id, and inference_id holds the inference id)
   *
   * working properties (pulled from dnn_info):
   * processed_frames, total_frames
   *
   * working/completed properties (pulled from dnn_info):
   * sequence_number
   * (sequence_number is also used as the success_count in the UI code)
   *
   * status values:
   * "starting", “working", “completed"
   */

  // the task_id is the deployed model id and the resource_id is the inference id.
  // this doesn't really match the rest of the SSEs, so change it up
  data.inference_id = data.resource_id;
  data.resource_id = data.task_id;
  data.task_id = data.inference_id;
  data.sub_msg = ['Action', 'Object'].includes(data.inference_type) ? data.inference_type.toLowerCase() : null;

  data.values = {
    processed_frames: makeFloat(data.processed_frames), // used by info
    total_frames: makeFloat(data.total_frames), // used by info
    success_count: makeFloat(data.sequence_number), // used by success
  };

  if (data.status === 'working' && data.values.total_frames) {
    // percent_complete isn't included in the SSE, so we need to set it here
    data.percent_complete = (data.values.processed_frames / data.values.total_frames) * 100;
    data.show_progress_bar = true;
  }

  if (data.status === 'completed') {
    data.sub_more_info = data.sub_msg;
  } else if (data.status === 'working' && data.values.total_frames) {
    data.sub_more_info = 'progress'; // needs processed_frames and total_frames
  }

  data.intl_id = 'depModel.inference';

  return data;
};

export const processorMap = {
  'import_files': processImportFilesMsg,
  'import_model': processImportModelMsg,
  'deploy_model': processDeployModelMsg,
  'inferencing': processInferencingMsg,
};
