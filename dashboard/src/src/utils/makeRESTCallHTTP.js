import http from 'http';

/**
 * Make a REST call using GET method and NodeJS builtin `http` module.
 * NB: A `https` module is also available for HTTPS addresses.
 * @param {string} path - address to load
 * @param {Object} parameters - GET parameters
 * @param resolve - function to call on success. Will receive JSON data.
 * @param reject - function to call on error. Will receive raw error object.
 */
export function makeRESTCallHTTP(path, parameters = {}, resolve, reject) {
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
