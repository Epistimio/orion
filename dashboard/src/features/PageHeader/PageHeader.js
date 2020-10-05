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
import './styles.scss';

import React from 'react';

import { ArrowLeft24 } from '@carbon/icons-react';
import { GpuSystemUsage } from '../Common';

const PageHeader = ({
  className,
  children,
  title,
  showBackButton,
  showGpuUsage,
  onClickBackButton,
}) => (
  <div className={'page-header' + (className ? (' ' + className) : '')}>
    <div className="page-header__page-title header-container__flexbox">
      <div className="header-title__container">
        {showBackButton
          && (
            <div
              tabIndex={0}
              role="button"
              className="header__back-button"
              onClick={onClickBackButton}
              onKeyPress={(e) => e.which === 13 && onClickBackButton()}>
              <ArrowLeft24 description="Return to previous page" />
            </div>
          )}
        {title}
        {children}
      </div>
      {showGpuUsage && <GpuSystemUsage />}
    </div>

  </div>
);


export default PageHeader;
