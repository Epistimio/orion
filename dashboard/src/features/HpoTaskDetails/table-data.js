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
import React from './node_modules/react';
import { Link } from './node_modules/react-router-dom';
import { FormattedMessage } from './node_modules/react-intl';
import { showTimestamp } from './node_modules/utils/helpers';

export const getHeaderList = () => {
  return [
    {
      key: 'id',
      header: <FormattedMessage id="table.id" />,
      decorator: (data) => {
        //return <Link to={`/hpotasks/${data}`} className="bx--link">{data}</Link>;
        return <Link to={`/overview`} className="bx--link">{data}</Link>;
      }
    },
    { key: 'params_authorName', header: <FormattedMessage id="table.author" /> },
    { key: 'params_experimentName', header: <FormattedMessage id="table.experimentName" /> },
    { key: 'revision', header: <FormattedMessage id="table.revision" /> },
    { key: 'startTime', header: <FormattedMessage id="table.startTime" />, decorator: (data) => showTimestamp(data) }
  ];
};

export const createSearchSortData = (notebooks) => {
  const headers = getHeaderList();
  let searchData = {};
  let sortData = {};
  notebooks && notebooks.allIds.forEach((notebookId) => {
    const notebook = notebooks.byId[notebookId];
    let searchStr = '';
    let sortObj = {};
    headers.forEach((header) => {
      const { key } = header;
      const value = notebook[key];

      if (value) {
        searchStr += ' ' + ((!header.ignoreDecoratorSearch && header.decorator) ? header.decorator(value) : value);
        sortObj[key] = (!header.ignoreDecoratorSort && header.sort) ? header.sort(value) : value;
      } else {
        sortObj[key] = '';
      }
    });
    searchData[notebookId] = searchStr.toLocaleLowerCase();
    sortData[notebookId] = sortObj;
  });
  return {
    searchData,
    sortData
  };
};
