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
import '../ImageCard/styles.scss';

import React from 'react';

import { Checkbox } from 'carbon-components-react';
import { ImageCard } from '../../Common';

const TableSelectRow = React.memo(({
  checked,
  id,
  resourceId,
  name,
  onChange,
  onClick,
  bgImgPath,
  showThumbnails
}) => (
  <td>
    {showThumbnails ? (
      <ImageCard
        className="controlled-data-table__image-card"
        checked={checked}
        resourceId={resourceId}
        forDataTable={true}
        bgImgPath={bgImgPath}
        onCardClick={onClick}
        onKeyClick={onClick}
        onCheckedChange={onClick}
      />
    ) : (
      <Checkbox
        className="controlled-data-table__row-checkbox"
        id={id}
        labelText={id}
        hideLabel={true}
        name={name}
        onChange={(checked) => onChange(resourceId, checked)}
        checked={checked}
      />
    )}
  </td>
));

export default TableSelectRow;
