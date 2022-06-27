import { makeRESTCallAxios as makeRESTCall } from './makeRESTCallAxios';

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
