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

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Link, withRouter } from 'react-router-dom';
import { injectIntl, FormattedMessage } from 'react-intl';

import {
  HeaderContainer,
  Header,
  HeaderName,
  HeaderNavigation,
  HeaderMenuButton,
  HeaderMenuItem,
  HeaderMenu,
  HeaderGlobalBar,
  HeaderGlobalAction,
  SideNav,
  SideNavItems,
  HeaderSideNavItems,
  SkipToContent
} from 'carbon-components-react';

import {
  Help20, ChevronDownGlyph,
  Notification20, NotificationFilled20,
  UserAvatar20
} from '@carbon/icons-react';

import CalendarSVG from 'images/nav_calendar.svg';

import { getHelpLink } from 'utils/helpers';
import { toggleNotificationCenter, HasActiveClosedSelector } from '../Notifications';

class Navigation extends Component {
  isCurrent(linkPath) {
    return this.props.location.pathname === linkPath ? 'page' : 'false';
  }

  createPageLink(endpoint, intlId) {
    return (
      <HeaderMenuItem
        element={Link}
        to={`/${endpoint}`}
        aria-current={this.isCurrent(`/${endpoint}`)}>
        <FormattedMessage id={intlId || `navigation.${endpoint}`} />
      </HeaderMenuItem>
    );
  }

  createDropdownItems = () => {
    return (
      <>
        {this.props.userId === 'admin'}
        {this.props.version && (
          <HeaderMenuItem className="dropdownDisabled-container">
            <span className="dropdownDisabledItem">
              <FormattedMessage id="dashboard.productVersion" values={{ versionNum: this.props.version }} />
            </span>
          </HeaderMenuItem>
        )}
      </>
    );
  };

  userDropdownItem = () => {
    return (
      <>
        <UserAvatar20 className="vision-navigation__user-icon" />
        <span className="vision-navigation__user-name">
          {this.props.userId}
        </span>
        <ChevronDownGlyph className="bx--header__menu-arrow" />
      </>
    );
  };

  render() {
    const { intl, notificationBarOpen, timeTrialMode, nonProdMode } = this.props;
    const isTimedEdition = nonProdMode || timeTrialMode;

    return (
      <HeaderContainer
        render={({ isSideNavExpanded, onClickSideNavExpand }) => (
          <Header aria-label="Orion Dashboard" className="app-header">
            <SkipToContent />
            <HeaderMenuButton
              aria-label="Open menu"
              onClick={onClickSideNavExpand}
              isActive={isSideNavExpanded}
            />
            <HeaderName element={Link} to="/" prefix={intl.formatMessage({ id: 'navigation.companyName' })}>
              <FormattedMessage id="navigation.productName" />
            </HeaderName>
            <HeaderNavigation aria-label="Orion Dashboard">
              {this.createPageLink('hpotasks')}
              {this.createPageLink('overview')}
              {this.createPageLink('detail')}
            </HeaderNavigation>
            <HeaderGlobalBar>
              <HeaderGlobalAction
                aria-label="Help"
                onClick={() => { window.open('https://orion.readthedocs.io/en/stable/', '_blank', 'noopener'); }}>
                <Help20 />
              </HeaderGlobalAction>
            </HeaderGlobalBar>
            <SideNav
              aria-label="Side navigation"
              expanded={isSideNavExpanded}
              isPersistent={false}>
              <SideNavItems>
                <HeaderSideNavItems>
                  <HeaderMenuItem element={Link} to="/hpotasks" onClick={onClickSideNavExpand}>
                    <FormattedMessage id="navigation.notebooks" />
                  </HeaderMenuItem>
                </HeaderSideNavItems>
              </SideNavItems>
            </SideNav>
          </Header>
        )}
      />
    );
  }
}

const mapStateToProps = state => {
  return {
    userId: state.auth.userId,
    authToken: state.auth.authToken,
    notificationBarOpen: state.notifications.notificationBarOpen,
    hasActiveClosedNotifications: HasActiveClosedSelector(state)
  };
};

export default injectIntl(withRouter(connect(mapStateToProps)(Navigation)));
