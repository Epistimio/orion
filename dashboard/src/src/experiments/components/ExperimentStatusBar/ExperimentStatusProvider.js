import { Backend, DEFAULT_BACKEND } from '../../../utils/queryServer';

/**
 * Helper class to provide and cache experiment status.
 */
class ExperimentStatusProvider {
  constructor() {
    this.backend = new Backend(DEFAULT_BACKEND);
    this.resolving = new Map();
    this.resolved = new Map();
  }

  get(experiment) {
    if (this.resolved.has(experiment)) {
      console.log(`Already resolved: ${experiment}`);
      return new Promise((resolve, reject) => {
        const resolution = this.resolved.get(experiment);
        if (resolution.error) {
          reject(resolution.error);
        } else {
          resolve(resolution.data);
        }
      });
    } else if (this.resolving.has(experiment)) {
      console.log(`Still resolving: ${experiment}`);
      return this.resolving.get(experiment);
    } else {
      console.log(`To resolve: ${experiment}`);
      this.resolving.set(
        experiment,
        new Promise((resolve, reject) => {
          this.backend
            .query(`experiments/status/${experiment}`)
            .then(status => {
              this.resolving.delete(experiment);
              this.resolved.set(experiment, { error: null, data: status });
              resolve(status);
            })
            .catch(error => {
              this.resolving.delete(experiment);
              this.resolved.set(experiment, { error, data: null });
              reject(error);
            });
        })
      );
      return this.resolving.get(experiment);
    }
  }
}

export const EXPERIMENT_STATUS_PROVIDER = new ExperimentStatusProvider();
