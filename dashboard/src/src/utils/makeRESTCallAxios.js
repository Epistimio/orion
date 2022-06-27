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
export function makeRESTCallAxios(path, parameters, resolve, reject) {
  const config = {
    method: 'get',
    url: path,
    responseType: 'json',
    responseEncoding: 'utf8',
    adapter: adapter,
    timeout: 300000,
  };
  if (Object.keys(parameters).length) config.params = parameters;
  axios(config)
    .then(response => resolve(response.data))
    .catch(reject);
}
