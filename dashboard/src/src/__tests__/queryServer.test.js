import { Backend, DEFAULT_BACKEND } from '../utils/queryServer';

test('test backend call', () => {
  expect.assertions(3);
  // Return promise to make sure jest will test async function
  const backend = new Backend(DEFAULT_BACKEND);
  return backend.query('').then(response => {
    expect(response.hasOwnProperty('orion')).toBeTruthy();
    expect(response.server).toBe('gunicorn');
    expect(response.database).toBe('PickledDB');
  });
});

test('test backend query experiments', () => {
  expect.assertions(8);
  const backend = new Backend(DEFAULT_BACKEND);
  return backend.query('experiments').then(response => {
    expect(response.length).toBe(111);
    const experiments = new Set(response.map(experiment => experiment.name));
    expect(experiments.has('2-dim-shape-exp')).toBeTruthy();
    expect(experiments.has('4-dim-cat-shape-exp')).toBeTruthy();
    expect(experiments.has('2-dim-exp')).toBeTruthy();
    expect(experiments.has('3-dim-cat-shape-exp')).toBeTruthy();
    expect(experiments.has('random-rosenbrock')).toBeTruthy();
    expect(experiments.has('tpe-rosenbrock')).toBeTruthy();
    expect(experiments.has('hyperband-cifar10')).toBeTruthy();
  });
});

test('test backend bad call', () => {
  expect.assertions(1);
  // Create backend with unreachable address.
  const backend = new Backend('http://localhost:1');
  return backend.query('').catch(error => {
    expect(error.message).toBe('connect ECONNREFUSED 127.0.0.1:1');
  });
});
