import axios from 'axios';
// Use adapter to prevent "cross origin forbidden" error in tests
// Ref (2022/02/01): https://github.com/axios/axios/issues/1418
import adapter from 'axios/lib/adapters/http';

/**
 * Make a REST call using GET method and axios module.
 * @param {string} path - address to load
 * @param {Object} parameters - GET parameters
 * @param resolve - function to call on success. Will receive JSON data.
 * @param reject - function to call on error. Will receive raw error object.
 */
function makeRESTCall(path, parameters, resolve, reject) {
  const config = {
    method: 'get',
    url: path,
    responseType: 'json',
    responseEncoding: 'utf8',
    adapter: adapter,
    timeout: 60000,
  };
  if (Object.keys(parameters).length) config.params = parameters;
  axios(config)
    .then(response => resolve(response.data))
    .catch(reject);
}

/** Wrapper class to call Orion Web API. */
export class Backend {
  /**
   * Create a Bckend instance.
   * @param {string} address - base URL of Orion server
   */
  constructor(address) {
    this.baseURL = address;
  }

  /**
   * query backend
   * @param path - path to load on Orion server. Will be appended to base URL.
   * @param params - query parameters.
   * @returns {Promise} - Promise with JSON data on success and raw error otherwise.
   */
  query(path, params = {}) {
    return new Promise((resolve, reject) => {
      const apiCall = `${this.baseURL}/${path}`;
      makeRESTCall(apiCall, params, resolve, reject);
    });
  }
}

/** Default address value. */
export const DEFAULT_BACKEND = 'http://127.0.0.1:8000';
