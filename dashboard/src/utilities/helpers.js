/* eslint-disable radix */
/* eslint-disable no-restricted-globals */
/* eslint-disable operator-assignment */
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
import normalize from './normalize';
import { store } from '../index';
import moment from 'moment';

export const createFilePath = (userId, file, datasetId) => {
  //filePath must be a relative URL to location of the application
  let filePath = `uploads/${userId}/datasets/${datasetId}/files/${file.file_name}`;
  return filePath;
};

// // TODO:mldlppc/tracker#2013 Remove need for constructing Thumbnail Path
export const createThumbnailPath = (userId, file, datasetId) => {
  //filePath must be a relative URL to location of the application
  let filePath = `uploads/${userId}/datasets/${datasetId}/thumbnails/`;

  let fileType = file.file_type;
  if (fileType === 'video') {
    filePath += file._id + '.jpg';
  } else {
    filePath += file.file_name;
  }

  return filePath;
};

export const modifyFileData = (data, datasetId) => {
  const currentState = store.getState();
  const { userId } = currentState.auth;
  // TODO it'd be nice to get this data from the backend instead of piecing it together here
  let videoFrameMap = {};
  let files = normalize(data, '_id');

  files.allIdsNoFrames = [];
  files.allIds.forEach(fileId => {
    const file = files.byId[fileId];
    file.filePath = createFilePath(file.owner || userId, file, datasetId);
    file.thumbnailPath = createThumbnailPath(file.owner || userId, file, datasetId);

    if (file.file_type === 'video' && !videoFrameMap[file._id]) {
      // keep track of this video so that we can handle it later
      videoFrameMap[file._id] = { id: file._id, categoryMap: {}, tagMap: {}, frames: [] };
      files.allIdsNoFrames.push(file._id);
    } else if (file.file_type === 'video-frame' && file.parent_id) {
      // video-frames aren't pushed to the sortedIdList for now, they'll be added when we iterate through the videos
      let videoObj = videoFrameMap[file.parent_id] || { id: file.parent_id, frames: [], categoryMap: {}, tagMap: {} };
      videoObj.frames.push(file);
      const catId = file.category_id || 'uncategorized';

      if (catId) {
        if (videoObj.categoryMap[catId]) {
          // increment this category's count
          videoObj.categoryMap[catId].category_count = videoObj.categoryMap[catId].category_count + 1;
        } else {
          // add this category to the array since it's not already there
          videoObj.categoryMap[catId] = {
            category_id: catId,
            category_name: file.category_name,
            category_count: 1,
          };
        }
      }

      if (file.tag_list && file.tag_list.length) {
        // add these tags to the array if they aren't already there
        file.tag_list.forEach(tagInfo => {
          if (videoObj.tagMap[tagInfo.tag_id]) {
            // increment this tag's count
            videoObj.tagMap[tagInfo.tag_id].tag_count = videoObj.tagMap[tagInfo.tag_id].tag_count + 1;
          } else {
            videoObj.tagMap[tagInfo.tag_id] = tagInfo;
          }
        });
      } else if (videoObj.tagMap.untagged) {
        // add this to the untagged list
        videoObj.tagMap.untagged.tag_count = videoObj.tagMap.untagged.tag_count + 1;
      } else {
        // untagged object hasn't been created yet, add it in
        videoObj.tagMap.untagged = {
          tag_id: 'untagged',
          tag_name: null,
          tag_count: 1,
        };
      }
      videoFrameMap[file.parent_id] = videoObj;
    } else {
      // push this image to the non-frame id list
      files.allIdsNoFrames.push(file._id);
    }
  });

  Object.values(videoFrameMap).forEach(video => {
    // videos don't have their own categories or objects, but save off which ones the frames are using
    const file = files.byId[video.id];
    video.frames.sort((frame1, frame2) => {
      return (
        (frame1.frame_info && !isNaN(parseInt(frame1.frame_info.time_offset)) ? +frame1.frame_info.time_offset : 0)
        - (frame2.frame_info && !isNaN(parseInt(frame2.frame_info.time_offset)) ? +frame2.frame_info.time_offset : 0)
      );
    });
    file.frames = normalize(video.frames, '_id');
    file.categories = Object.values(video.categoryMap);
    file.tags = Object.values(video.tagMap);
  });

  return files;
};

export const createConfig = progressFunc => ({
  headers: { 'content-type': 'multipart/form-data' },
  onUploadProgress: progressEvent => {
    let percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
    if (percent >= 100) {
      progressFunc(100, false);
      //Start the timer once we have completed upload
      setTimeout(() => {
        progressFunc(0, true); //After 1 second, set progress to 0
      }, 1000);
    } else {
      progressFunc(percent, false);
    }
  },
});

export const createVideoTimeStamp = (timeMilliSeconds, doTrim) => {
  let hours = Math.floor(timeMilliSeconds / 3600000);
  let minutes = Math.floor((timeMilliSeconds - hours * 3600000) / 60000);
  let seconds = Math.floor((timeMilliSeconds - hours * 3600000 - minutes * 60000) / 1000);
  let milliseconds = ((timeMilliSeconds - hours * 3600000 - minutes * 60000 - seconds * 1000) / 1000).toFixed(3);

  if (hours < 10) {
    hours = doTrim ? hours : '0' + hours;
  }
  if (minutes < 10) {
    minutes = doTrim && !hours ? minutes : '0' + minutes;
  }
  if (seconds < 10) {
    seconds = '0' + seconds;
  }
  if (milliseconds === '0.000') {
    milliseconds = '000';
  } else {
    milliseconds = ('00' + Math.floor(milliseconds * 1000)).slice(-3);
  }

  if (doTrim) {
    return (
      (hours !== 0 ? hours + ':' : '')
      + (minutes !== 0 || hours !== 0 ? minutes + ':' : '')
      + seconds
      + '.'
      + milliseconds
    );
  }

  if (hours === '00') {
    return minutes + ':' + seconds + '.' + milliseconds;
  }
  return hours + ':' + minutes + ':' + seconds + '.' + milliseconds;
};

export const convertTimeStampToMS = timeStamp => {
  // Used for user input that is in time-stamp form that needs to be sent
  // to the backend is millisecond form
  let milliseconds = 0;
  let [seconds, minutes, hours] = timeStamp.split(':').reverse();
  milliseconds += minutes ? Number(minutes) * 60000 : 0;
  milliseconds += hours ? Number(hours) * 3600000 : 0;
  if (seconds) {
    // look for milliseconds to break the seconds down further
    let [s, ms] = seconds.split('.');
    ms = ms && ms.substring(0, 3); // make sure the ms value is a max of 3 numbers
    milliseconds += ms ? Number(ms) * Number('1' + '0'.repeat(3 - ms.length)) : 0;
    milliseconds += s ? Number(s) * 1000 : 0;
  }

  return milliseconds;
};

export const acceleratorTypes = {
  GPU: 'GPU',
  GPU_TENSORRT: 'GPU_TENSORRT',
  CPU: 'CPU',
  FPGA_XFDNN_8_BIT: 'FPGA_XFDNN_8_BIT',
  FPGA_XFDNN_16_BIT: 'FPGA_XFDNN_16_BIT',
};

export const segmentationColors = [
  'rgba(230, 25, 75, x)',
  'rgba(60, 180, 75, x)',
  'rgba(255, 225, 25, x)',
  'rgba(67, 99, 216, x)',
  'rgba(245, 130, 49, x)',
  'rgba(145, 30, 180, x)',
  'rgba(66, 212, 244, x)',
  'rgba(240, 50, 230, x)',
  'rgba(191, 239, 69, x)',
  'rgba(250, 190, 190, x)',
  'rgba(70, 153, 144, x)',
  'rgba(230, 190, 255, x)',
  'rgba(154, 99, 36, x)',
  'rgba(255, 250, 200, x)',
  'rgba(128, 0, 0, x)',
  'rgba(170, 255, 195, x)',
  'rgba(128, 128, 0, x)',
  'rgba(255, 216, 177, x)',
  'rgba(0, 0, 117, x)',
  'rgba(169, 169, 169, x)',
];

export const augmentTypes = [
  'gaussian_blur',
  'motion_blur',
  'sharpness',
  'crop',
  'color',
  'rotation',
  'noise',
  'flip_vertical',
  'flip_horizontal',
];

export const getHelpLink = (filename, vrfm, lang = 'en') => {
  const currentState = store.getState();
  const version = vrfm;

  // knowledge center uses 3 digit versions, truncate version string
  const vrf = version && version.substring(0, version.length - 2);

  return `https://www.ibm.com/support/knowledgecenter/${lang}/SSRU69_${vrf}/base/${filename}`;
};

export const marketplaceLink = 'https://www.ibm.com/us-en/marketplace/ibm-powerai-vision';

export const trialLink = 'https://ibm.biz/powerai-vision-trial-info';

// Helper method to...
// 1. initially sort the image and video files to help tie an augmented image to it's parent image.
// 2. sort video frames to help tie an augmented frame to it's parent video frame.
export const createInitialSortedFiles = files => {
  if (files && files.allIdsNoFrames) {
    let initialSortedFiles = [];
    let augmentedFiles = [];
    let initialSortedFrames = {};
    let augmentedFrames = {};

    files.allIdsNoFrames.forEach(id => {
      const file = files.byId[id];
      if (file.augment_method) {
        // move augmented image into a list so the image can be tied to its parent image
        augmentedFiles.push(id);
      } else {
        // if not an augmented image, put file in sorted list
        initialSortedFiles.push(id);
      }

      // sort video frames
      if (file.file_type === 'video') {
        if (file.frames && file.frames.allIds && file.frames.allIds.length) {
          let sortedFramesList = [];
          let augmentedFramesList = [];
          file.frames.allIds.forEach(frameId => {
            const frame = file.frames.byId[frameId];
            if (frame.augment_method) {
              // move augmented frame into a list so the frame can be tied to its parent frame
              augmentedFramesList.push(frameId);
            } else {
              // if not an augmented frame, put video frame in sorted list of frames
              sortedFramesList.push(frameId);
            }
          });
          initialSortedFrames[id] = sortedFramesList;
          augmentedFrames[id] = augmentedFramesList;
        }
      }
    });

    // add the matching augmented files to their parent image in the initally sorted files list
    augmentedFiles.forEach(id => {
      const file = files.byId[id];
      let index = initialSortedFiles.indexOf(file.original_file_id);
      if (index < 0) {
        initialSortedFiles.push(id);
      } else {
        initialSortedFiles.splice(index + 1, 0, id);
      }
    });
    // add the matching augmented frames to their parent frame
    Object.keys(augmentedFrames).forEach(videoId => {
      let augmentList = augmentedFrames[videoId];
      let framesList = initialSortedFrames[videoId];
      augmentList.forEach(id => {
        const file = files.byId[id];
        let index = framesList.indexOf(file.original_file_id);
        if (index < 0) {
          framesList.push(id);
        } else {
          framesList.splice(index + 1, 0, id);
        }
      });
      initialSortedFrames[videoId] = framesList;
    });

    return {
      initialSortedFiles,
      initialSortedFrames,
    };
  }
  return [];
};

export const showTimestamp = (timestamp) => {
  return moment(timestamp).format('L, LT');
};

// String "ncpus=1,mem=8192" to a map object
export const resourceRequestStrToObj = (str) => {
  let obj = {};
  if(!str) return obj;
  const requests= str.split(",");
  requests.forEach(request => {
    const items = request.split("=");
    obj[items[0]] = items[1] * 1.0;
  });
  return obj;
}

// Change and sum mss job resource request strings to map object
export const getResourceRequestInfo = (task0Resreq, task12nResreq) => {
  let info = {
    cpu : 0,
    gpu : 0,
    mem : 0,
  };
  const task0Req = resourceRequestStrToObj(task0Resreq);
  const task12nReq = resourceRequestStrToObj(task12nResreq);
  if(task0Req.ncpus)    info.cpu += task0Req.ncpus;
  if(task12nReq.ncpus)  info.cpu += task12nReq.ncpus;
  if(task0Req.ngpus)    info.gpu += task0Req.ngpus;
  if(task12nReq.ngpus)  info.gpu += task12nReq.ngpus;
  if(task0Req.mem)      info.mem += task0Req.mem;
  if(task12nReq.mem)    info.mem += task12nReq.mem;
  return info;
}

export const getDuration = (start, end) => {
  let durationn = "0.0";
  let tmpDuration;
  if(start!=0 && end!=0){
    tmpDuration = ((end-start)/1000.0)/60.0;
  }else if(start!=0){
    const currentDate = new Date();
    const currentTime = currentDate.getTime(); 
    tmpDuration = ((currentTime-start)/1000.0)/60.0;
  }
  if (tmpDuration >= 0.0) {
    durationn = parseFloat(tmpDuration).toFixed(1);
  } else {
    durationn = "0.0";
  }
  return durationn;
}

// Add and get the history path for navigation.
export let historyPath = [];
export const maxHistoryNumber = 5 ;

// call this in the Navigation.js
export const addHistoryPath = (path) => {
  historyPath.unshift(path);
  if(historyPath.length>maxHistoryNumber) historyPath.pop();
}

// call this in any component which need go back to history page.
// index = 0: is the current page, 1..n: is the history page. (n<5)
export const getHistoryPath = (index) => {
  if(index<0 || index>=maxHistoryNumber || index>=historyPath.length) return "";
  if(historyPath.length==0) return "";
  return historyPath[index];
}
