/** React context to store backend (orion server) address, with a default value. */
import React from 'react';
import { DEFAULT_BACKEND } from '../utils/queryServer';

/** React context. */
export const BackendContext = React.createContext({
  address: DEFAULT_BACKEND,
  // We use React context to store selected experiment accross pages
  experiment: null,
});
