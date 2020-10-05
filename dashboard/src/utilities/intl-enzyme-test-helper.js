/* eslint-disable import/no-extraneous-dependencies */
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

/**
 * Components using the react-intl module require access to the intl context.
 * This is not available when mounting single components in Enzyme.
 * These helper functions aim to address that and wrap a valid, English-locale intl context around them.
 * See https://github.com/yahoo/react-intl/wiki/Testing-with-React-Intl#enzyme for more information.
 */

import React from 'react';
import { IntlProvider, intlShape } from 'react-intl';
import { mount, shallow } from 'enzyme';

// Create the IntlProvider to retrieve context for wrapping around.
const intlProvider = new IntlProvider({ locale: 'en' }, {});
const { intl } = intlProvider.getChildContext();

/**
 * When using React-Intl `injectIntl` on components, props.intl is required.
 */
const nodeWithIntlProp = node => {
  return React.cloneElement(node, { intl });
};

export const shallowWithIntl = (node, { context, ...additionalOptions } = {}) => {
  return shallow(nodeWithIntlProp(node), {
    context: { ...context, intl },
    ...additionalOptions,
  });
};

export const mountWithIntl = (node, { context, childContextTypes, ...additionalOptions } = {}) => {
  return mount(nodeWithIntlProp(node), {
    context: { ...context, intl },
    childContextTypes: { intl: intlShape, ...childContextTypes },
    ...additionalOptions,
  });
};
