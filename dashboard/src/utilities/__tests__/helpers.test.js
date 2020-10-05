/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018                                           */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import { createFilePath, createThumbnailPath, createVideoTimeStamp } from '../helpers';

test('file path given a file and dataset ID', () => {
  expect(createFilePath({file_name: "file1.jpg"}, "dataset1")).toBe("uploads/admin/datasets/dataset1/files/file1.jpg");
});

test('thumbnail path given a file and dataset ID', () => {
  const file = {file_name: "file1.jpg", file_type: "video", _id: "1"}
  expect(createThumbnailPath(file, "dataset1")).toBe("uploads/admin/datasets/dataset1/thumbnails/1.jpg");

  file.file_type = "image";
  expect(createThumbnailPath(file, "dataset1")).toBe("uploads/admin/datasets/dataset1/thumbnails/file1.jpg");
});

test('Create video timestamp given time in milliseconds', () => {
  // check when no milliseconds in timestamp
  expect(createVideoTimeStamp(640000)).toBe("10:40.00");
  // check milliseconds in timestamp
  expect(createVideoTimeStamp(123456)).toBe("02:03.46");
  // check hours in timestamp
  expect(createVideoTimeStamp(38987654)).toBe("10:49:47.65");
});
