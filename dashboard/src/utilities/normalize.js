/* eslint-disable prefer-destructuring */
/* eslint-disable camelcase */
/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2020                                           */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */

/* Redux recommends "normalizing" data so that the byId property contains an object
 * that uses IDs as keys, and the actual data as the value at that key. This simplifies
 * data lookups significantly since you don't need to search the whole array to find a
 * single piece of data that matches a given ID. Additionally, allIds is used to hold
 * an array of all possible IDs, which is used for determining order.
 */
import { segmentationColors, createVideoTimeStamp } from 'utils/helpers';
import { compareAsc } from 'utils/sorting';

const normalize = (data, idProp = '_id') => {
  let allIds = [];
  let byId = {};

  if (data) {
    data.forEach(dataObj => {
      allIds.push(dataObj[idProp]);
      byId[dataObj[idProp]] = dataObj;
    });
  }

  return {
    byId,
    allIds,
    count: allIds.length,
  };
};

// Used to normalize all the object-labels in a single dataset
export const normalizeLabels = labels => {
  // we can fetch all the labels for each file in a dataset at once,
  // so put them in a format that we can easily look them up by file
  let allIds = [];
  let byId = {};
  let fileIds = [];
  let labelsByFileId = {};
  let hasAutoLabels = false;

  labels && labels.forEach(labelObj => {
    const { _id, file_id, generate_type } = labelObj;

    if (file_id) {
      allIds.push(_id);
      byId[_id] = labelObj;

      if (!hasAutoLabels && generate_type === 'auto') {
        hasAutoLabels = true; // this dataset has at least one auto label
      }

      if (labelsByFileId[file_id]) {
        // append to this file's list
        labelsByFileId[file_id].hasAutoLabels = labelsByFileId[file_id].hasAutoLabels || generate_type === 'auto';
        labelsByFileId[file_id].labels.push(labelObj);
      } else {
        // reached a new file
        labelsByFileId[file_id] = {
          labels: [labelObj],
          hasAutoLabels: generate_type === 'auto',
        };
        fileIds.push(file_id);
      }
    }
  });

  return {
    objectLabels: { byId, allIds },
    fileLabels: {
      byId: labelsByFileId,
      allIds: fileIds,
    },
    hasAutoLabels,
  };
};

// Used to normalize action tag and action labels for action detection
export const normalizeActions = (datasetActions = [], fileActions = []) => {
  let { byId: tagsById, allIds: tagIds } = normalize(datasetActions);
  let labelIds = [];
  let labelsById = {};
  let labelsByParentId = {};

  // remove colors that look like the background color of the video progress bar
  let colorList = segmentationColors.filter(item => item !== 'rgba(67, 99, 216, x)' && item !== 'rgba(0, 0, 117, x)');

  // sort dataset actions from oldest to newest, then by name if created_at is equal
  tagIds
    .sort((id1, id2) => compareAsc(tagsById[id1], tagsById[id2], 'created_at', 'name'))
    .forEach((tagId, index) => {
      // assign each dataset action a color
      let colorIndex = index % colorList.length;
      tagsById[tagId].color = colorList[colorIndex];
    });

  fileActions.forEach(dataObj => {
    const { _id, tag_id } = dataObj;
    if (!tag_id || !tagsById[tag_id]) {
      // this tag was somehow created incorrectly, just ignore it for now
      return;
    }

    if (labelsByParentId[tag_id]) {
      labelsByParentId[tag_id].push(_id);
    } else {
      labelsByParentId[tag_id] = [_id];
    }

    // the actual timestamp will be displayed to the user instead of ms; save it off in
    // the data object so we don't have to keep re-calculating for filtering and displaying
    labelsById[_id] = {
      ...dataObj,
      start_timestamp: dataObj.start_time !== null ? createVideoTimeStamp(dataObj.start_time, true) : null,
      end_timestamp: dataObj.end_time !== null ? createVideoTimeStamp(dataObj.end_time, true) : null,
      color: tagsById[tag_id].color,
      tag_name: dataObj.tag_name || tagsById[tag_id].name || '',
    };

    labelIds.push(_id);
  });

  // the labels can be viewed either nested under their parent action-tags,
  // or on their own. byParentId will be used to sort fileActions within their parents,
  // and allIds will be used to sort fileActions as a whole list.
  return {
    datasetActions: {
      byId: tagsById,
      allIds: tagIds,
    },
    fileActions: {
      byId: labelsById,
      byParentId: labelsByParentId,
      allIds: labelIds,
    },
  };
};

// Used to normalize all the /tags in a dataset
export const normalizeDatasetObjects = objects => {
  // we don't need to use any previous data since objects will always be in the same order
  // this is mainly just used if we get a new dataset object, or one is deleted
  let objectsById = {};
  let objectAllIdsCopy = [...objects.allIds]; // used to sort the objects in the right order

  // sort the objects so that the oldest objects are first in the list.
  // then, convert to a format where the IDs are "A", "B", etc instead
  // of the tag ID
  objectAllIdsCopy
    .sort((objId1, objId2) => compareAsc(objects.byId[objId1], objects.byId[objId2], 'created_at', 'name'))
    .forEach(objId => {
      objectsById[objId] = { ...objects.byId[objId] };
      objectsById[objId].labelIds = [];
    });

  return {
    // list of all object possibilities
    objectList: {
      byId: objectsById,
      allIds: objectAllIdsCopy,
    },
  };
};

// Used to merge a dataset's /tags and a single file's /object-labels
// so that the labels are mapped to their parent tag
export const normalizeObjects = (objects, labels) => {
  let labelsByObjId = {};
  let objectsById = {};
  let labelsById = {};
  let autoLabeledIds = [];
  const objectAllIdsCopy = [...objects.allIds]; // used to sort the objects in the right order
  const labelAllIdsCopy = [...labels.allIds];

  // match individual labels up with their parent object using the tag_id
  labelAllIdsCopy.forEach(labelId => {
    const label = labels.byId[labelId];
    labelsById[labelId] = label;
    if (label.tag_id && objects.byId[label.tag_id]) {
      // this is a valid label with a valid parent
      if (labelsByObjId[label.tag_id]) {
        labelsByObjId[label.tag_id].push(label);
      } else {
        labelsByObjId[label.tag_id] = [label];
      }
    }
    if (label.generate_type && label.generate_type === 'auto') {
      autoLabeledIds.push(labelId);
    }
  });

  // sort the objects so that the oldest objects are first in the list.
  // then, convert to a format where the IDs are "A", "B", etc instead
  // of the tag ID
  objectAllIdsCopy
    .sort((objId1, objId2) => compareAsc(objects.byId[objId1], objects.byId[objId2], 'created_at', 'name'))
    .forEach(objId => {
      const labels = labelsByObjId[objId];

      objectsById[objId] = { ...objects.byId[objId] };
      objectsById[objId].labelIds = [];

      labels && labels.forEach(label => {
        if (label._id) {
          objectsById[objId].labelIds.push(label._id);
        }
      });
    });

  return {
    // list of all object possibilities
    objectList: {
      byId: objectsById,
      allIds: objectAllIdsCopy,
    },
    // list of actual bounding boxes and polygons on the image
    labelList: {
      byId: labelsById,
      allIds: labelAllIdsCopy,
    },
    autoLabeledIds,
  };
};

// Used to normalize the results returned on an COD inference
export const normalizeInferenceLabels = (labels, confthre) => {
  let idObjectMap = {};
  let objectsById = {};
  let labelsById = {};
  let objectIds = [];
  let labelIds = [];
  let colorIndex = 0;

  // filter out the object detection results with a lower confidence than the user determined threshold
  const filteredResult = labels.filter(label => {
    return Number(label.confidence) >= confthre;
  });

  // go through each label (boxes and polygons) on the image
  // and save it off in a { labelName: label } format
  filteredResult.forEach(label => {
    const currentObj = idObjectMap[label.label];
    if (currentObj) {
      currentObj.labels.push(label);
      currentObj.count += 1;
      currentObj.average += (label.confidence - currentObj.average) / currentObj.count;
    } else {
      idObjectMap[label.label] = {
        name: label.label,
        count: 1,
        average: label.confidence,
        colors: segmentationColors[colorIndex],
        labels: [label],
      };
      // index through the approved list of colors
      colorIndex = (colorIndex + 1) % segmentationColors.length;
    }
  });

  // Convert to a format where the IDs are "A", "B", etc instead
  // of the actual label name
  Object.keys(idObjectMap).forEach((name, index) => {
    // replace names with letters
    const objectData = idObjectMap[name];
    let objLetter = ((index % 26) + 10).toString(36).toLocaleUpperCase();
    // once we hit "Z", we should continue on with "AA", "BB", etc
    objLetter = objLetter.repeat(index >= 26 ? (Math.floor(index / 26) + 1) : 1);
    objectsById[objLetter] = objectData;
    objectsById[objLetter].labelIds = [];
    if (objectData.labels.length === 1) {
      // if there is only one label, just stick with the base
      // letter for the label ID instead of adding "1" to it
      objectsById[objLetter].labelIds.push(objLetter);
      labelsById[objLetter] = objectData.labels[0];
      labelsById[objLetter].colors = objectData.colors;
      labelIds.push(objLetter);
    } else {
      // otherwise, go through each label and increment the ID number each time
      objectData.labels.forEach((label, index) => {
        const newId = `${objLetter}${index + 1}`;
        objectsById[objLetter].labelIds.push(newId);
        labelsById[newId] = label;
        labelsById[newId].colors = objectData.colors;
        labelIds.push(newId);
      });
    }

    objectIds.push(objLetter);
  });

  return {
    // list of all object possibilities
    objectList: {
      byId: objectsById,
      allIds: objectIds,
    },
    // list of actual bounding boxes and polygons on the image
    labelList: {
      byId: labelsById,
      allIds: labelIds,
    },
  };
};

export default normalize;
