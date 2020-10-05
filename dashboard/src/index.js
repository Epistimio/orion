
import './index.scss';

/* Node module imports */
import React from 'react';
import ReactDOM from 'react-dom';

import { createStore, applyMiddleware, compose } from 'redux';
import { Provider } from 'react-redux';
import ReduxThunk from 'redux-thunk';

import { addLocaleData, IntlProvider } from 'react-intl';
import en from 'react-intl/locale-data/en';

import { EventSourcePolyfill } from 'event-source-polyfill';

/* Internal module imports */
import rootReducer from './rootReducer';
import messages from './locales/messages';
import App from './App';

global.EventSourcePolyfill = EventSourcePolyfill;
addLocaleData([...en]);

// Setting locale to English-only,
// once translation is supported, update this to fetch browser locale
const locale = 'en';

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
export const store = createStore(rootReducer, composeEnhancers(applyMiddleware(ReduxThunk)));

ReactDOM.render(
  <IntlProvider locale={locale} messages={messages[locale]}>
    <Provider store={store}>
      <App />
    </Provider>
  </IntlProvider>,
  document.getElementById('root')
);
