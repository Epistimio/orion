import http from 'http';

/**
 * Make a REST call using GET method and NodeJS builtin `http` module.
 * NB: A `https` module is also available for HTTPS addresses.
 * @param {string} path - address to load
 * @param {Object} parameters - GET parameters
 * @param resolve - function to call on success. Will receive JSON data.
 * @param reject - function to call on error. Will receive raw error object.
 */
function makeRESTCall(path, parameters = {}, resolve, reject) {
  let url = path;
  const keys = Object.keys(parameters);
  if (keys.length) {
    url += '?';
    for (let key of keys) {
      url += `${key}=${encodeURI(parameters[key])}`;
    }
  }
  http
    .get(url, resp => {
      let data = '';
      resp.on('data', chunk => {
        data += chunk;
      });
      resp.on('end', () => {
        resolve(JSON.parse(data));
      });
    })
    .on('error', err => {
      reject(err);
    });
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
