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
import { processEvent } from './processSSEEvent';

const processBackgroundTasks = bgTask => {
  let data = {
    ...bgTask,
    resource_id: bgTask.resource_id,
    task_id: bgTask.task_id || bgTask._id,
    is_bgtask: true,
    created_at: bgTask.created_at || bgTask.updated_at || Date.now(),
  };

  if (bgTask.hasOwnProperty('total_item_count')) {
    // SSEs use total_file_count instead
    data.total_file_count = bgTask.total_item_count;
  }
  if (!bgTask.hasOwnProperty('success_count') && bgTask.hasOwnProperty('completed_items')) {
    // SSEs use success_count instead of completed_items
    data.success_count = bgTask.completed_items;
  }

  if (bgTask.event_type === 'auto_label') {
    data.resource_type = bgTask.operation;
    data.resource_id = bgTask.file_id;
  } 

  processEvent({ data }, data.event_type);
};

export default processBackgroundTasks;
