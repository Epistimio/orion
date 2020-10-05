/* eslint-disable max-len */
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
export const sortStates = {
  NONE: 'NONE',
  DESC: 'DESC',
  ASC: 'ASC',
};

export const initialSortState = sortStates.NONE;

// Used by ControlledDataTable to get the next sort direction when a header key
// is clicked. It will iterate through the sort states depending on if the header
// is already sorted or not.
export const getNextSortDirection = (prevHeader, header, prevState) => {
  // If the previous header is equivalent to the current header, we know that we
  // have to derive the next sort state from the previous sort state
  if (prevHeader === header) {
    // When transitioning, we know that the sequence of states is as follows:
    // NONE -> ASC -> DESC -> NONE
    if (prevState === 'NONE') {
      return sortStates.ASC;
    }
    if (prevState === 'ASC') {
      return sortStates.DESC;
    }
    return sortStates.NONE;
  }
  // Otherwise, we have selected a new header and need to start off by sorting
  // in descending order by default
  return sortStates.ASC;
};

/**
 * Use the built-in `localeCompare` function available on strings to compare.
 *
 * @param {string} a
 * @param {string} b
 * @param {string} locale
 * @returns {number}
 */
export const compareStrings = (a, b, locale = 'en') => {
  return a.localeCompare(b, locale, { numeric: true });
};

/**
 * Compare two primitives to determine which comes first. Initially, this method
 * will try and figure out if both entries are the same type. If so, it will
 * apply the default sort algorithm for those types. Otherwise, it defaults to a
 * string conversion.
 *
 * @param {number|string} a
 * @param {number|string} b
 * @param {string} locale
 * @returns {number}
 */
export const compare = (a, b, locale = 'en') => {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }

  if (typeof a === 'string' && typeof b === 'string') {
    return compareStrings(a, b, locale);
  }

  return compareStrings('' + a, '' + b, locale);
};

/**
 * Default implementation of how we sort rows internally. The idea behind this
 * implementation is to use the given list of row ids to look up the cells in
 * the row by the given sortHeaderKey. We then use the value of these cells and pipe them
 * into our local `compareStrings` method, including the locale where
 * appropriate.
 *
 * @param {Object} config
 * @param {Array[string]} config.idList array of all the row ids
 * @param {Object} config.objectByIdMap object containing a mapping of resource id to
 * resource
 * @param {string} config.sortDirection the sort direction used to determine the
 * order the comparison is called in
 * @param {string} config.sortHeaderKey the header key that we use to lookup the cell
 * @param {string} [config.locale] optional locale used in the comparison function
 * @returns {Array[string]} array of sorted rowIds
 */
export const sortIds = ({ idList, objectByIdMap, sortDirection, sortHeaderKey, locale = 'en' }) => idList.slice().sort((a, b) => {
  const objA = objectByIdMap[a];
  const objB = objectByIdMap[b];
  if (sortDirection === sortStates.ASC) {
    return compare(objA[sortHeaderKey], objB[sortHeaderKey], locale);
  }
  return compare(objB[sortHeaderKey], objA[sortHeaderKey], locale);
});

/**
 * Used within a sorting function to compare 2 objects.
 * Example: objects.sort((obj1, obj2) => compareAsc(obj1, obj2, "created_at", "name"));
 *
 * Note: This does not use a localeCompare so that the order is consistent
 *       for every single user, regardless of location
 *
 * @param {Object} Object1 that contains the supplied properties
 * @param {Object} Object2 that contains the supplied properties
 * @param {string} The property name that the objects should be sorted by
 * @param {string} Optional. Second property to sort by in the result of a tie
 * @returns {number} For compareAsc: -1 if obj1 comes before obj2, 0 if same, 1 if obj1 comes after obj2
 *                   For compareDesc: -1 if obj1 comes after obj2, 0 if same, 1 if obj1 comes before obj2
 */
export const compareAsc = (obj1 = {}, obj2 = {}, sortProp1 = '', sortProp2 = '') => {
  const diff = (obj1[sortProp1] > obj2[sortProp1]) - (obj1[sortProp1] < obj2[sortProp1]);
  if (diff === 0 && sortProp2) {
    // if diff is 0, sort by second prop
    return (obj1[sortProp2] > obj2[sortProp2]) - (obj1[sortProp2] < obj2[sortProp2]);
  }
  return diff;
};

export const compareDesc = (obj1 = {}, obj2 = {}, sortProp1 = '', sortProp2 = '') => {
  const diff = (obj2[sortProp1] > obj1[sortProp1]) - (obj2[sortProp1] < obj1[sortProp1]);
  if (diff === 0 && sortProp2) {
    // if diff is 0, sort by second prop
    return (obj2[sortProp2] > obj1[sortProp2]) - (obj2[sortProp2] < obj1[sortProp2]);
  }
  return diff;
};
