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
import { FormattedMessage } from 'react-intl';
import { Button } from 'carbon-components-react';

import UsersSVG from 'images/DashboardIllustration-users.svg'
import { PageHeader } from '../PageHeader';

const ImageContainer = ({ svg, intlId, linkurl }) => {
  return (
    <div className="dashboard__image-container">
      <svg className="dashboard__image" 
           viewBox={svg.viewBox} 
           fillRule="evenodd"
           onClick={() => { window.open(linkurl)}}
      >
        <use xlinkHref={`#${svg.id}`} />
      </svg>
      <div className="dashboard__image-text">
        <FormattedMessage id={intlId} />
      </div>
    </div>
  );
};

const Dashboard = (props) => (
  <div className="dashboard">
    <PageHeader
      title={<FormattedMessage id="dashboard.welcomeMessage" />}
    />

    <div className="body-container">
      <div className="dashboard__images-container">
        <ImageContainer intlId="dashboard.userManagement" svg={UsersSVG} linkurl='#/hpotasks'/>
      </div>
      <Button
        className="dashboard__btn--get-started"
        onClick={() => {
          props.history.push({
            pathname: '/detail'
          });
        }}>
        <FormattedMessage id="btn.getStarted" />
      </Button>      
    </div>
  </div>
);

export default Dashboard;
