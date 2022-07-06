import axios from 'axios';
// Use adapter to prevent "cross origin forbidden" error in tests
// Ref (2022/02/01): https://github.com/axios/axios/issues/1418
import adapter from 'axios/lib/adapters/http';

/**
 * Api call task.
 * Used by task queue to chain queries and execute them sequentially
 * instead of in parallel.
 */
class Task {
  static nextID = 0;

  /**
   * Initialize
   * @param {Backend} backend - backend to use for API call
   * @param {string} path - path to use for API call
   * @param {Object} parameters - parameters to use for API call
   * @param {Task} previousTask - Task to wait for before running current one
   */
  constructor(backend, path, parameters, previousTask) {
    this.taskID = Task.nextID++;
    this.previousTask = previousTask;
    this.backend = backend;
    this.path = path;
    this.parameters = parameters;
    this.promise = null;
    this.innerRun = this.innerRun.bind(this);
    this.run = this.run.bind(this);
  }
  toString() {
    return `[${this.taskID}] ${this.backend.baseURL} / ${
      this.path
    } ? ${JSON.stringify(this.parameters)}`;
  }

  /**
   * Task run function (private)
   * Wait for previous task if necessary, then run API call for current task
   */
  async innerRun() {
    try {
      if (this.previousTask) {
        console.log(`Waiting ... ${this.previousTask}\nCurrent ... ${this}`);
        await this.previousTask.promise;
      }
    } finally {
      console.log(`... Ended   ${this.previousTask}\nCurrent ... ${this}`);
      delete this.previousTask;
      return await this.backend.innerQuery(this.path, this.parameters);
    }
  }

  /**
   * Task run function (public)
   */
  run() {
    this.promise = this.innerRun();
    return this.promise;
  }
}

/**
 * Task queue
 * Use a chained list to run API calls sequentially instead of in parallel
 * Used to prevent backend from receiving too many requests in few time.
 * Helpful if backend has few resources for heavy computation.
 */
class TaskQueue {
  constructor() {
    this.previousTask = null;
  }
  query(backend, path, parameters) {
    const task = new Task(backend, path, parameters, this.previousTask);
    this.previousTask = task;
    return task.run();
  }
}

const TASK_QUEUE = new TaskQueue();

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
    timeout: 400000,
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
   * query backend (private method, to be called by TaskQueue)
   * @param path - path to load on Orion server. Will be appended to base URL.
   * @param params - query parameters.
   * @returns {Promise} - Promise with JSON data on success and raw error otherwise.
   */
  innerQuery(path, params = {}) {
    return new Promise((resolve, reject) => {
      const apiCall = `${this.baseURL}/${path}`;
      makeRESTCall(apiCall, params, resolve, reject);
    });
  }

  /**
   * query backend (public method, add query to task queue)
   * See this.innerQuery() for more details
   */
  query(path, params = {}) {
    return TASK_QUEUE.query(this, path, params);
  }
}

/** Default address value. */
export const DEFAULT_BACKEND = 'http://127.0.0.1:8000';
